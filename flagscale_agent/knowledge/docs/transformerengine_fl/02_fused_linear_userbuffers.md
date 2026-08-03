# 第二章：Fused Linear Ops & Userbuffers 通信-计算重叠

## 1. 概述与设计动机

TransformerEngine-FL 的核心性能创新：将 Linear 层的 GEMM 计算与 Tensor/Sequence Parallel 通信（AllGather、ReduceScatter）在硬件层面深度融合。通过 NCCL Userbuffers 注册内存机制实现零拷贝通信，使 NVLink 带宽传输与 SM 计算完全并行。

**核心问题**: 大模型 TP/SP 训练中，每个 Linear 层前后都有 AllGather 或 ReduceScatter 通信。朴素实现中通信和计算串行执行，浪费了 GPU 的并行能力。

**解决方案**: 将通信操作直接融合到 GEMM kernel 的调度中，通过 CUDA stream 并发和分块 pipeline 实现计算-通信重叠。

---

## 2. 源文件定位

| 文件路径 | 行数 | 核心职责 |
|---------|------|---------|
| `pytorch/ops/fused/userbuffers_forward_linear.py` | 448 | Forward: AG+GEMM / GEMM+RS 融合 |
| `pytorch/ops/fused/userbuffers_backward_linear.py` | 669 | Backward: dgrad/wgrad + AG/RS 融合 |
| `pytorch/module/linear.py` | 1670 | 高层 Linear API (Column/RowParallel) |
| `pytorch/module/base.py` | 1693 | `fill_userbuffers_buffer_for_all_gather` 核心函数 |
| `pytorch/cpp_extensions/gemm.py` | ~200 | `general_gemm` — 统一 GEMM 接口 (支持 UB 参数) |
| `pytorch/userbuffers.py` | ~500 | CommOverlapHelper / UB 注册管理 |

---

## 3. 架构层次

```
┌──────────────────────────────────────────────────────────────┐
│  用户 API                                                      │
│  ColumnParallelLinear / RowParallelLinear                      │
│  (module/linear.py)                                           │
├──────────────────────────────────────────────────────────────┤
│  Fused Operation 层                                           │
│  UserbuffersForwardLinear  (ops/fused/ L36-448)               │
│  UserbuffersBackwardLinear (ops/fused/ L39-669)               │
├──────────────────────────────────────────────────────────────┤
│  基础原语                                                      │
│  fill_userbuffers_buffer_for_all_gather (base.py L500-600)    │
│  general_gemm (cpp_extensions/gemm.py L75+)                   │
├──────────────────────────────────────────────────────────────┤
│  NCCL Userbuffers 层                                          │
│  CommOverlapHelper → ncclCommRegister() → 零拷贝 AG/RS       │
├──────────────────────────────────────────────────────────────┤
│  硬件层: cuBLAS SM ←→ NVLink/NIC (并行执行)                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Forward 通信重叠 — UserbuffersForwardLinear

### 4.1 类结构 (L36-84)

```python
class UserbuffersForwardLinear(FusedOperation):
    """Linear forward with comm overlap via NCCL userbuffers"""
    
    def __init__(self, *, linear: BasicLinear, bias: Optional[Bias],
                 reduce_scatter: Optional[ReduceScatter]):
        # 组合基础 op: linear + bias + reduce_scatter
        op_idxs = {"linear": 0, "bias": None, "reduce_scatter": None}
        ops = [linear]
        ...
        # 确定 TP 模式
        if reduce_scatter is None:
            self.tensor_parallel_mode = linear.tensor_parallel_mode  # "column"
        else:
            self.tensor_parallel_mode = "row"  # RS 存在 → RowParallel
```

**关键设计**: 通过组合模式将 `BasicLinear` + `Bias` + `ReduceScatter` 三个基础 op 融合为一个 fused operation，使得通信可以和前一个/后一个 op 重叠。

### 4.2 _functional_forward 核心逻辑 (L86-277)

#### ColumnParallel 模式 (AllGather + GEMM)

```python
# L192-224: AllGather 准备
ub_comm = get_ub(ub_comm_name + "_fprop", with_quantized_compute)
with_ub_all_gather = (tensor_parallel_mode == "column")  # L194
ub_type = CommOverlapType.AG  # L196

# L201-216: 量化 + 填充 UB buffer
if with_ub_all_gather:
    if input_quantizer is not None:
        input_quantizer.set_usage(rowwise=True, columnwise=False)  # L208: AG后无需列存储
        x_local = input_quantizer(x_local)  # FP8 量化
    x, x_local = fill_userbuffers_buffer_for_all_gather(
        ub_comm, x_local, input_quantizer, tensor_parallel_group
    )
```

#### fill_userbuffers_buffer_for_all_gather 详解 (base.py L500-600)

```python
def fill_userbuffers_buffer_for_all_gather(comm, local_tensor, quantizer, process_group):
    """填充 UB buffer 的 local shard, 返回 (全局tensor视图, 本地tensor视图)"""
    
    process_group_size = torch.distributed.get_world_size(process_group)
    global_shape = list(local_shape)
    global_shape[0] *= process_group_size  # 第0维扩大 TP 倍
    
    # FP8 路径 (L541-561):
    if isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
        comm.copy_into_buffer(local_tensor._data, local_chunk=True)  # 零拷贝写入注册内存
        global_tensor_data = comm.get_buffer(shape=global_shape)      # 获取完整 buffer 视图
        global_tensor = Float8TensorStorage(
            data=global_tensor_data,
            fp8_scale_inv=local_tensor._scale_inv,  # scale 共享 (per-tensor scaling)
            fp8_dtype=local_tensor._fp8_dtype,
            ...
        )
        return global_tensor, local_tensor
```

**关键**: `copy_into_buffer(local_chunk=True)` 将本 rank 的数据写入 UB buffer 的对应位置。AG 操作只需将其他 rank 的 shard 写入 buffer 的其余位置——由于 buffer 已注册到 NCCL，可以直接 RDMA 写入，无需 staging copy。

#### GEMM 执行 (L242-257)

```python
# L243-253: GEMM 通过 general_gemm 统一接口执行
gemm_output, *_, reduce_scatter_output = general_gemm(
    w,                              # weight (可能是 FP8)
    x,                              # full input (AG 后的完整 tensor)
    out_dtype=dtype,
    quantization_params=output_quantizer,
    bias=bias,
    use_split_accumulator=_2X_ACC_FPROP,
    ub=ub_comm,                     # ← Userbuffers 通信对象
    ub_type=ub_type,                # AG 或 RS
    extra_output=reduce_scatter_output,  # RS 输出 buffer
)
```

**`ub` 参数传入 general_gemm 的意义**: cuBLAS 内部可以在 GEMM 分块执行时，将已完成的块通过 UB 异步发送/接收，实现 chunk-level pipeline overlap。

### 4.3 ColumnParallel vs RowParallel 时序对比

**ColumnParallel Forward (AG overlap GEMM):**
```
时间轴 ──────────────────────────────────────────────────────→

Compute Stream: [FP8量化 x_local] → [等待AG完成] → [GEMM: x_full @ W^T] → [输出 y]
                                      ↑                        
Comm Stream:    [AG: 收集各rank的x_local] ─────────────────────
                ↑                          ↑
            copy_into_buffer           AG complete (event signal)
```

**RowParallel Forward (GEMM overlap RS):**
```
时间轴 ──────────────────────────────────────────────────────→

Compute Stream: [GEMM: x @ W^T] ─────→ [产出 y_full] → [下一层可开始]
                                                ↓
Comm Stream:                              [RS: y_full → y_local]
                                              ↑ 与下一层计算 overlap
```

### 4.4 ub_comm_name 命名约定

```python
# L193: get_ub(ub_comm_name + "_fprop", with_quantized_compute)
# 约定命名:
#   "qkv_fprop"  — QKV 投影的前向
#   "proj_fprop" — attention output projection 前向
#   "fc1_fprop"  — FFN 第一层前向
#   "fc2_fprop"  — FFN 第二层前向
```

每个 comm_name 对应独立的 UB buffer 对象，避免不同层的通信互相干扰。

---

## 5. Backward 通信重叠 — UserbuffersBackwardLinear

### 5.1 _functional_backward 核心逻辑 (L91-497)

反向传播需要计算两个梯度:
- **dgrad**: `dx = dy @ W` (对输入的梯度)
- **wgrad**: `dW = x^T @ dy` (对权重的梯度)

#### 5.1.1 dgrad GEMM + AG (L332-401)

```python
# ColumnParallel backward: dy 需要 AllGather (因为前向 RS 了 output)
if with_dgrad_all_gather_dy:  # L340
    dy, _ = fill_userbuffers_buffer_for_all_gather(
        ub_comm_dgrad, dy_local, grad_output_quantizer, tensor_parallel_group
    )

# dgrad GEMM: dx = dy_full @ W_columnwise
dx, *_, dx_local = general_gemm(
    w, dy,
    out_dtype=dtype,
    layout="NN",
    ub=ub_comm_dgrad,
    ub_type=ub_type_dgrad,         # AG 或 RS
    extra_output=reduce_scatter_output,  # 如果需要 RS dx
)
```

#### 5.1.2 wgrad GEMM + Bulk Overlap (L403-483)

```python
# wgrad GEMM: dW = x^T @ dy
# wgrad 与 dgrad 的 reduce-scatter 重叠执行
dw, *_ = general_gemm(
    x, dy,
    out_dtype=dw_dtype,
    layout="NT",           # x 需要转置
    accumulate=accumulate_into_grad_weight,
    ub=ub_comm_wgrad,      # wgrad 使用独立的 UB communicator
    ub_type=ub_type_wgrad,
    bulk_overlap=with_bulk_overlap,  # ← 关键: 标记为 bulk overlap 模式
)
```

**bulk_overlap** 的含义: wgrad GEMM 执行期间，dgrad 的 ReduceScatter 在通信 stream 上并行运行。因为 wgrad 是 compute-bound (大矩阵乘法)，可以完全隐藏 dgrad RS 的通信。

#### 5.1.3 MXFP8 特殊路径 (L408-450)

```python
# L408-413: MXFP8 不支持 pipelined AG-wgrad overlap
if tensor_parallel_mode == "row" and isinstance(grad_output_quantizer, MXFP8Quantizer):
    # 无法重用 dgrad 的 gathered dy (row-scaled → column-scaled 不兼容)
    # 退化为: 利用 dgrad GEMM 的通信 stream 显式 overlap AG
    dgrad_send_stream, dgrad_recv_stream = ub_comm_dgrad.get_communication_stream()
    with torch.cuda.stream(dgrad_send_stream):
        dy, _ = fill_userbuffers_buffer_for_all_gather(ub_obj_overlap_wgrad, ...)
```

### 5.2 Backward 完整时序图

**ColumnParallel Backward (最复杂场景):**
```
时间轴 ──────────────────────────────────────────────────────────────→

Compute:  [量化dy] → [AG_dy wait] → [dgrad: dy_full@W] → [wgrad: x^T@dy_full]
                                                          ↑ overlap ↓
Comm 1:   [AG dy_local → dy_full] ────────────────────────
Comm 2:                                    [RS dx_full → dx_local] ────────
                                                          ↑ wgrad 期间执行
```

**RowParallel Backward:**
```
时间轴 ──────────────────────────────────────────────────────────────→

Compute:  [量化dy] → [dgrad: dy@W^T] → [AG x wait] → [wgrad: x_full^T@dy]
                                         ↑              ↑ overlap ↓
Comm 1:              [AG x_local → x_full] ─────────────
Comm 2:                                                  [RS dgrad → local]
```

---

## 6. Userbuffers 底层机制详解

### 6.1 NCCL Registered Memory 原理

```python
class CommOverlapHelper:
    """管理 NCCL userbuffer 生命周期"""
    
    # 初始化流程:
    # 1. torch.cuda.memory.allocate(...) — 分配 GPU 全局内存
    # 2. ncclCommRegister(buffer_ptr, size) — 注册到 NCCL
    #    → NCCL 获得 buffer 的 GDR (GPU Direct RDMA) 映射
    #    → 后续 AG/RS 直接在注册区域上操作，无需 staging copy
    # 3. 提供 copy_into_buffer() / get_buffer() 接口
```

### 6.2 零拷贝 vs 传统模式对比

**传统 NCCL AllGather:**
```
GPU 0 data → [copy to NCCL internal buffer] → [NVLink send] → ...
             ↑ 额外 memcpy                   → [NVLink recv] → [copy to user buffer]
                                                                ↑ 额外 memcpy
```

**Userbuffers AllGather:**
```
GPU 0 data ────────────────→ [NVLink send from registered buffer] → ...
(已在注册区)                 → [NVLink recv into registered buffer]
                               ↑ 直接写入，无中间拷贝
```

### 6.3 性能分析

| 对比维度 | 传统 NCCL | Userbuffers | 收益 |
|---------|-----------|-------------|------|
| memcpy 次数 | 2 (in+out staging) | 0 | 节省 ~10-20% 通信时间 |
| SM 占用 | copy kernel 占 SM | 无 copy SM 需求 | SM 全给 GEMM |
| 延迟 | kernel launch overhead × 2 | 减少 2 次 launch | 降低尾延迟 |
| 内存 | 额外 staging buffer | 共享注册 buffer | 节省显存 |
| 限制 | 无 | buffer 大小固定,需预分配 | 灵活性降低 |

---

## 7. general_gemm 统一接口

### 7.1 函数签名 (cpp_extensions/gemm.py L75+)

```python
def general_gemm(
    A, B,
    out_dtype,
    *,
    quantization_params=None,  # 输出量化器
    bias=None,
    layout="NN",               # "NN", "NT", "TN"
    accumulate=False,          # 累加到现有输出
    use_split_accumulator=False,  # FP8 split accumulator (精度)
    ub=None,                   # Userbuffers comm object (启用重叠)
    ub_type=None,              # CommOverlapType.AG / CommOverlapType.RS
    extra_output=None,         # RS 输出 buffer
    bulk_overlap=False,        # bulk overlap 模式
    grad=False,                # 是否反向 pass
    ...
) -> tuple:
```

**设计精妙之处**: `ub` 参数使得同一个 GEMM 调用既能普通执行，也能自动触发通信重叠。调用者无需关心底层 pipeline 分块逻辑——全部封装在 `general_gemm` + cuBLAS plugin 内部。

### 7.2 通信-GEMM Pipeline 执行模式

当 `ub` 非 None 时，cuBLAS 内部执行分块 pipeline:

```
AG + GEMM Pipeline (num_chunks=4):
┌───────────────────────────────────────────────────────────┐
│ Step 1: AG chunk[0]      | ---                            │
│ Step 2: AG chunk[1]      | GEMM chunk[0]                 │
│ Step 3: AG chunk[2]      | GEMM chunk[1]                 │
│ Step 4: AG chunk[3]      | GEMM chunk[2]                 │
│ Step 5: ---              | GEMM chunk[3]                  │
└───────────────────────────────────────────────────────────┘
理想情况下: T_total ≈ max(T_AG, T_GEMM) + T_chunk (pipeline startup)
```

---

## 8. 高层 API 集成 (module/linear.py)

### 8.1 ColumnParallelLinear 路径选择

```python
class ColumnParallelLinear(TransformerEngineBaseModule):
    def forward(self, input):
        # 判断是否启用 UB overlap
        if self._userbuffers_options is not None:
            # 路径 A: UserbuffersForwardLinear (融合 AG+GEMM)
            # 通过 FusedOperation pipeline 自动调度
            ...
        else:
            # 路径 B: 显式 AllGather → BasicLinear GEMM
            if self.sequence_parallel:
                input = all_gather(input, tp_group)  # 阻塞通信
            output = F.linear(input, weight, bias)
```

### 8.2 Userbuffers 启用条件

```python
# 必须同时满足:
# 1. TP > 1 (否则无通信可重叠)
# 2. SP = True (AG/RS 发生在 sequence 维度)
# 3. UB 环境变量/配置启用: NVTE_UB_OVERLAP=1
# 4. NCCL 版本支持 user registration API
```

---

## 9. FP8 与 Userbuffers 的协同

### 9.1 FP8 AllGather 特殊处理

FP8 tensor 的 AllGather 需要同步 `scale_inv`:

```python
# base.py L541-561:
if isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
    comm.copy_into_buffer(local_tensor._data, local_chunk=True)  # 只 AG 数据
    global_tensor = Float8TensorStorage(
        data=global_tensor_data,
        fp8_scale_inv=local_tensor._scale_inv,  # scale 不需要 AG (per-tensor 共享)
        ...
    )
```

**关键洞察**: DelayedScaling 模式下，同一层所有 TP rank 使用相同的 scale (通过 amax all-reduce 保证)。因此 AG 只需传输 uint8 数据，scale_inv 直接复用本地值。

### 9.2 数据类型对通信量的影响

| 精度 | 每元素字节 | AG 通信量 (seq=4K, hidden=4096, TP=8) | 相比 BF16 |
|------|-----------|--------------------------------------|-----------|
| BF16 | 2 | 4096 × 4096 × 2 = 32 MB | 基准 |
| FP8 | 1 | 4096 × 4096 × 1 = 16 MB | 50% ↓ |
| MXFP8 | 1 + scale | ~16.5 MB (含 block scales) | ~48% ↓ |

FP8 训练天然将通信量减半，与 Userbuffers 叠加可获得显著加速。

---

## 10. Backward 量化器传递与存储管理

### 10.1 Forward 保存的反向所需信息 (L354-367)

```python
# fuser_forward 中保存状态:
linear_op_ctx.save_for_backward(x_local, w)  # 量化后的 input 和 weight
linear_op_ctx.with_quantized_compute = with_quantized_compute
linear_op_ctx.input_quantizer = input_quantizer
linear_op_ctx.weight_quantizer = weight_quantizer
linear_op_ctx.grad_output_quantizer = grad_output_quantizer
linear_op_ctx.grad_input_quantizer = grad_input_quantizer
```

**设计意图**: 反向时需要前向的量化参数来正确处理 FP8 tensor 的 dequantize/re-quantize。weight 的 columnwise 数据 (转置) 直接用于 dgrad GEMM，无需重新量化。

### 10.2 weight 的 rowwise/columnwise 使用切换 (L260-264)

```python
# Forward 后准备反向的 weight:
if input_requires_grad:
    if w is not weight and with_quantized_compute and is_quantized_tensor(w):
        w.update_usage(rowwise_usage=False, columnwise_usage=True)  # 反向只需列存储
```

---

## 11. 性能模型与通信量化

### 11.1 单层通信量分析

设 `s`=序列长度×batch, `h`=hidden_dim, `t`=TP_size, `b`=元素字节数:

| 操作 | 通信量 | 发生位置 |
|------|--------|---------|
| ColumnParallel Forward AG | `s × h × b × (t-1)/t` | 收集完整 input |
| RowParallel Forward RS | `s × h × b × (t-1)/t` | 分发 output |
| ColumnParallel Backward AG (dy) | `s × h × b × (t-1)/t` | 收集完整 grad_output |
| ColumnParallel Backward RS (dx) | `s × h × b × (t-1)/t` | 分发 grad_input |
| Wgrad AG (input for wgrad) | `s × h × b × (t-1)/t` | (若需重新收集) |

### 11.2 Overlap 理论加速比

```
无重叠: T_layer = T_gemm + T_comm (串行)
有重叠: T_layer = max(T_gemm, T_comm) + T_startup

加速比 = (T_gemm + T_comm) / max(T_gemm, T_comm)

# H100 估算 (hidden=8192, TP=8, seq×batch=8192):
#   T_gemm ≈ 8192^2 × 8192 / 989e12 ≈ 0.56 ms (BF16 TFLOPS)
#   T_comm ≈ 8192 × 8192 × 2 / 450e9 ≈ 0.30 ms (NVLink 单向)
#   加速比 ≈ (0.56+0.30)/0.56 = 1.54x (理论上限)
```

### 11.3 Overlap 效率的影响因素

| 因素 | 影响 | 优化建议 |
|------|------|---------|
| GEMM 大小 | GEMM 越大，overlap 隐藏通信越充分 | 增大 micro_batch_size |
| TP 组内带宽 | NVLink > PCIe > RDMA | 优先使用 NVSwitch |
| Pipeline chunk 数 | 过多增加 startup，过少 overlap 不充分 | 通常 4-8 chunks |
| FP8 vs BF16 | FP8 通信量减半，更容易被 GEMM 隐藏 | FP8 训练天然优势 |

---

## 12. 设计决策总结

| 设计选择 | 方案 | 替代方案 | 权衡理由 |
|---------|------|---------|---------|
| Fused Operation 模式 | 组合 Linear+Bias+RS | 逐 Op 手动重叠 | 编译器风格自动融合，用户透明 |
| 每层独立 UB buffer | comm_name 隔离 | 共享全局 buffer | 避免层间数据竞争，允许并发通信 |
| AG 后 columnwise=False | 不存列格式 | 保留双存储 | AG 后的 full tensor 是临时的，节省显存 |
| wgrad bulk overlap | wgrad 隐藏 dgrad RS | 分块 pipeline | wgrad 矩阵大,duration 足够隐藏通信 |
| MXFP8 退化路径 | 显式 stream overlap | 统一 pipeline | MXFP8 block scale 不兼容行列转换 |
| scale 不参与 AG | 本地复用 | all-gather scale | DelayedScaling 保证全局 scale 一致 |

---

## 13. 调优指南

### 13.1 启用配置

```yaml
# Megatron-LM-FL 配置
model:
  tp_comm_overlap: true
  # 可选细粒度控制:
  tp_comm_overlap_cfg:
    qkv_fprop: {ub_overlap: "ag"}      # QKV forward: AG overlap
    proj_fprop: {ub_overlap: "rs"}     # Proj forward: RS overlap
    fc1_fprop: {ub_overlap: "ag"}      # FFN1 forward: AG overlap
    fc2_fprop: {ub_overlap: "rs"}      # FFN2 forward: RS overlap

# 环境变量
NVTE_UB_OVERLAP=1                      # 总开关
NVTE_UB_FP8=1                          # FP8 buffer 模式
```

### 13.2 问题排查

| 症状 | 可能原因 | 排查 |
|------|---------|------|
| UB 不生效 (无加速) | NCCL 版本不支持 register API | 检查 NCCL ≥ 2.19 |
| CUDA OOM | UB buffer 预分配过大 | 减小 buffer size 或 chunk 数 |
| 数值不一致 | FP8 AG 后 scale 不匹配 | 验证 amax all-reduce 配置 |
| 性能退化 | GEMM 太小 (通信 > 计算) | 增大 batch 或降低 TP |
