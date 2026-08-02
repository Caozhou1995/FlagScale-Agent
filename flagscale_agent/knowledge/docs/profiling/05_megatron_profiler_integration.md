# Chapter 05: Megatron-LM-FL Profiler集成与分析 深度分析

## 1. 设计架构

**WHY Megatron需要自己的profiler集成**: 分布式训练的profiling不同于单卡 —
需要按rank选择性采集、与PP/TP/DP的通信关联、与训练step精确对齐。
Megatron提供了step-aware的wrapper层。

## 2. 双模式机制

### 2.1 nsys模式 (use_nsys_profiler=True)

```python
# 源码: training.py L2554-2590
# Megatron通过NVTX Range标记让nsys精确采集:

if args.profile and args.use_nsys_profiler:
    # 在profile_step_start到profile_step_end之间:
    with nsys_nvtx_context(
        "train_step", iteration, args
    ):
        # 整个train_step被NVTX包裹
        # nsys可以按NVTX filter
    
# NVTX标记层次 (从nsys timeline可见):
# train_step/
#   ├── forward_pass/
#   │   ├── layer_0/
#   │   │   ├── attention/
#   │   │   └── mlp/
#   │   └── layer_N/
#   ├── backward_pass/
#   └── optimizer_step/
```

### 2.2 PyTorch Profiler模式

```python
# 源码: training.py L2957-3020
# 使用torch.profiler.profile + schedule:

schedule = torch.profiler.schedule(
    wait=max(args.profile_step_start - 1, 0),
    warmup=1,
    active=args.profile_step_end - args.profile_step_start,
    repeat=1,
)

# 输出到 tensorboard_dir/../torch_profile/
# 每个rank生成独立trace文件
```

### 2.3 两种模式对比

```
┌──────────────┬─────────────────────┬──────────────────────────┐
│ 维度          │ nsys模式             │ PyTorch Profiler模式      │
├──────────────┼─────────────────────┼──────────────────────────┤
│ 采集粒度      │ GPU kernel级         │ Op/Module级              │
│ 可视化        │ nsys-ui GUI          │ TensorBoard/Chrome Trace │
│ 通信分析      │ ✅ NCCL timeline清晰 │ ⚠️ 只看到kernel名        │
│ CPU分析       │ ✅ 系统调用/线程      │ ✅ Python调用栈          │
│ 开销          │ <5%                  │ 5-15%                    │
│ 适用场景      │ timeline/gap分析      │ Op统计/内存/FLOPS        │
│ 多rank        │ 手动合并nsys         │ TB自动显示多rank         │
└──────────────┴─────────────────────┴──────────────────────────┘
```

## 3. Profile参数详解

### 3.1 配置参数

```python
# 源码: config/common_config.py L28-67 (已验证)

# 基础控制:
--profile                    # 总开关
--profile-step-start 10      # 第10步开始 (0-based)
--profile-step-end 12        # 第12步结束 (exclusive)
--profile-ranks 0 7          # 只采集rank 0和7

# PyTorch Profiler专用:
--use-pytorch-profiler       # 启用 (不设则用nsys模式)
--pytorch-profiler-collect-shapes    # tensor shape
--pytorch-profiler-collect-callstack # Python stack  
--pytorch-profiler-collect-chakra    # Chakra ET

# nsys专用:
--use-nsys-profiler          # 启用nsys NVTX模式

# WHY按rank采集:
# TP内各rank工作负载相同, 只需profile一个
# PP各stage不同, 需要选择不同stage的rank
# 例如 8GPU TP4PP2: rank 0是PP stage0, rank 7是PP stage1
```

### 3.2 FlagScale Config映射

```yaml
# FlagScale YAML config中对应:
train:
  model:
    profile: true
    profile_step_start: 10
    profile_step_end: 12
    profile_ranks: [0, 7]
    use_pytorch_profiler: true
    pytorch_profiler_collect_shapes: true
```

## 4. Timing与Metrics

### 4.1 内置Timer系统

```python
# Megatron自带细粒度timer, 不需要外部profiler:
# 源码: training.py training_log()

# 输出示例 (每N步打印):
# elapsed time per iteration (ms): 245.3
#   forward: 85.2 / backward: 142.1 / optimizer: 18.0
#   forward-compute: 72.1 / forward-recv: 13.1
#   backward-compute: 115.0 / backward-send-forward-recv: 27.1
#   layernorm-grads-all-reduce: 0.3
#   embedding-grads-all-reduce: 0.1
#   all-grads-reduce-scatter: 12.5
#   optimizer-inner: 15.0 / optimizer-allgather: 3.0

# 关键timer含义:
# forward-compute: 纯前向计算(不含PP通信)
# forward-recv: PP接收activation的等待
# backward-send-forward-recv: 1F1B调度中的send+recv
# all-grads-reduce-scatter: DP梯度通信
```

### 4.2 通信Overlap分析

```
通过nsys timeline可判断overlap效果:

理想状态 (通信被计算隐藏):
GPU0: [=== forward L0 ===][=== forward L1 ===]
NCCL: [---- allreduce ----]  ← 与forward重叠

不理想 (通信暴露):
GPU0: [=== forward ===][  idle  ][=== next ===]
NCCL:                  [allreduce]  ← 暴露在critical path

WHY看不到overlap:
- bucket_size太大 → 整个backward结束才发通信
- 单stream执行 → 没有compute/comm stream分离
- PP bubble → 结构性idle无法隐藏
```

## 5. 实战profiling工作流

### 5.1 完整流程

```bash
# Step 1: 快速定位 — Megatron内置timer
# 看training log里的时间分解:
# forward 85ms, backward 142ms → backward占比高, 正常

# Step 2: nsys全局timeline
nsys profile --trace=cuda,nvtx,osrt \
  -o train_timeline \
  --force-overwrite \
  torchrun --nproc_per_node=8 \
    python train.py --profile --profile-step-start 5 \
    --profile-step-end 7 --use-nsys-profiler

# Step 3: 打开nsys GUI分析
# 看: forward/backward/optimizer时间分布
# 看: GPU idle gap (黄色区域)
# 看: NCCL kernel与compute的overlap

# Step 4: 发现可疑kernel后, 用ncu深入
ncu --kernel-name "可疑kernel名" \
    --set full --launch-count 3 \
    python single_gpu_forward.py

# Step 5: 根据ncu结果优化
# → 调整tile/block配置
# → 使用融合kernel
# → 优化内存访问pattern
```

### 5.2 PP气泡分析

```
PP Schedule profiling (1F1B为例):

nsys timeline (4 PP stages, 4 microbatches):
Stage0: [F0][F1][F2][F3][B3][B2][B1][B0]
Stage1:     [F0][F1][F2][F3][B3][B2][B1][B0]
Stage2:         [F0][F1][F2][F3][B3][B2][B1][B0]  
Stage3:             [F0][F1][F2][F3][B3][B2][B1][B0]

气泡 = Stage0在B3之前的等待时间
理论气泡率 = (pp-1) / (microbatches + pp-1)

WHY profiling PP:
- 验证实际气泡是否符合理论
- 发现stage间不均衡(某stage forward比其他慢)
- 优化: 增大num_microbatches减少气泡率
```

## 6. 常见性能模式识别

```
模式1: GPU Idle Gap
  nsys看到两个kernel之间有空隙
  原因: CPU launch开销、Python GIL、sync操作
  
模式2: 小kernel碎片化  
  nsys看到大量<10μs的小kernel
  原因: 未融合的element-wise操作
  解决: 使用torch.compile或手动融合
  
模式3: 通信暴露
  NCCL kernel在critical path上
  原因: 没有overlap、bucket太大
  解决: 调小bucket_size、使用async grad allreduce
  
模式4: 内存bound GEMM
  ncu显示GEMM的memory SOL > compute SOL
  原因: 小batch/小shape
  解决: 增大batch或padding到对齐边界
```

## 7. 总结

```
Profiling工具链使用优先级:
┌──────────────────────────────────────────────────────────┐
│ 1. Megatron Timer → 每次训练都有, 快速判断大方向         │
│ 2. nsys           → 发现timeline问题(gap/overlap/idle)  │
│ 3. PyTorch Prof   → Op级统计, 快速找到最慢Op            │
│ 4. NCU            → 单kernel深度分析, 最后手段           │
│ 5. Memory Viz     → OOM/碎片化专项                      │
└──────────────────────────────────────────────────────────┘
```
