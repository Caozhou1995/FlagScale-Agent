# PyTorch Distributed 源码深度分析 — 第4章：DeviceMesh 与 DTensor

## 1. 设计动机

### 1.1 WHY DeviceMesh？

```
多维并行的组合爆炸问题:
────────────────────────────────────
传统方式: 手动创建 process group
  TP=4, PP=2, DP=4 → 需要手动算 ranks:
  tp_group = new_group([0,1,2,3])    # 重复 8 次
  dp_group = new_group([0,4,8,...])   # 重复 4 次
  pp_group = new_group([0,32])        # 重复 16 次
  
  问题: rank 计算容易出错, 难以维护和扩展

DeviceMesh: 多维网格抽象
  mesh = DeviceMesh("cuda", [[0,1,2,3],[4,5,6,7],...], 
                    mesh_dim_names=("dp","tp"))
  tp_group = mesh.get_group("tp")    # 自动计算
  dp_group = mesh.get_group("dp")    # 自动计算
```

### 1.2 在分布式训练栈中的位置

```
┌─────────────────────────────────────────────┐
│  用户代码 / Training Framework               │
├─────────────────────────────────────────────┤
│  DTensor (分布式 Tensor, 声明式分片)          │
│  Placement: Shard(dim), Replicate, Partial  │
├─────────────────────────────────────────────┤
│  DeviceMesh (多维设备网格)                    │
│  mesh_dim_names=("dp","tp","pp")            │
├─────────────────────────────────────────────┤
│  ProcessGroup (torch.distributed)            │
├─────────────────────────────────────────────┤
│  NCCL / Hardware                             │
└─────────────────────────────────────────────┘
```

## 2. DeviceMesh API

### 2.1 创建网格

```python
# torch/distributed/device_mesh.py
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# 方式1: 直接创建 (需要明确的 rank 布局)
mesh_2d = DeviceMesh("cuda", [[0,1,2,3],[4,5,6,7]], 
                     mesh_dim_names=("dp","tp"))

# 方式2: init_device_mesh (自动推断布局)
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
# world_size=8, dp=2, tp=4
# 自动构造: [[0,1,2,3],[4,5,6,7]]

# 方式3: 3D 网格
mesh_3d = init_device_mesh("cuda", (2, 2, 4), 
                           mesh_dim_names=("dp", "pp", "tp"))
```

### 2.2 获取子网格/进程组

```python
# 获取进程组:
tp_group = mesh["tp"]               # 返回子 DeviceMesh
dp_pg = mesh.get_group("dp")        # 返回 ProcessGroup

# 获取当前 rank 在某维度的坐标:
tp_rank = mesh.get_local_rank("tp")  # 当前进程的 TP rank
dp_rank = mesh.get_local_rank("dp")  # 当前进程的 DP rank

# 多维切片:
tp_mesh = mesh["tp"]  # 1D mesh, 只含 TP 维度
# 用于 FSDP: FSDP(model, device_mesh=mesh["dp"])
# 用于 TP: parallelize_module(model, mesh["tp"], plan)
```

## 3. DTensor — 分布式 Tensor

### 3.1 核心概念

```
DTensor = 全局逻辑 Tensor + Placement 描述 + DeviceMesh

Placement 类型:
┌─────────────────────────────────────────────────────┐
│ Shard(dim)   → tensor 在 mesh_dim 上按 dim 切分     │
│ Replicate    → tensor 在 mesh_dim 上完整复制         │
│ Partial(op)  → tensor 在 mesh_dim 上是 partial sum  │
│               (需要 reduce 才能得到正确值)            │
└─────────────────────────────────────────────────────┘

例子 (2D mesh: dp=2, tp=4):
  参数 W [4096, 16384]:
    Placement: [Replicate, Shard(1)]
    含义: DP 维度复制, TP 维度按列切分
    每个 GPU 持有: [4096, 4096] (16384/4)
```

### 3.2 DTensor 运算规则

```
DTensor 算子分派 (redistribute):
──────────────────────────────────
输入: X [Shard(0)] @ W [Shard(1)]  (矩阵乘)
输出: Y [Partial(SUM)]

原因: X 按行切, W 按列切 → 每个 rank 算的是 partial product
      需要 AllReduce 得到完整结果

Redistribute 通信:
  Partial → Replicate:  AllReduce
  Partial → Shard(dim): ReduceScatter
  Shard → Replicate:    AllGather
  Replicate → Shard:    本地 slice (无通信)
```

### 3.3 WHY DTensor?

```
WHY DTensor 而非手动分片?
─────────────────────────
1. 语义清晰: placement 声明 tensor 如何分布
2. 自动通信: 算子根据 input/output placement 自动插入 collective
3. 可组合: 与 FSDP2、TP、torch.compile 集成
4. Checkpoint 兼容: DTensor state_dict 可以变 world_size 加载

实际应用:
  Megatron 用手动 scatter/gather → 高性能但代码复杂
  DTensor 用声明式 → 代码简洁但有少量开销
```

## 4. Tensor Parallel via DTensor

```python
# torch/distributed/tensor/parallel/
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

# TP 并行化:
plan = {
    "attention.wq": ColwiseParallel(),    # Q 按列切: [H, H/tp]
    "attention.wk": ColwiseParallel(),    # K 按列切
    "attention.wv": ColwiseParallel(),    # V 按列切
    "attention.wo": RowwiseParallel(),    # O 按行切: [H/tp, H]
    "ffn.w1": ColwiseParallel(),          # FFN gate 按列切
    "ffn.w2": RowwiseParallel(),          # FFN down 按行切
}
parallelize_module(model, mesh["tp"], plan)

# 底层: 将 nn.Linear.weight 替换为 DTensor
# ColwiseParallel: weight → DTensor([Shard(0)]) 按输出维切
# RowwiseParallel: weight → DTensor([Shard(1)]) 按输入维切
```

## 5. 2D 并行 (FSDP + TP)

```python
# 标准 2D 并行配置:
mesh = init_device_mesh("cuda", (dp_size, tp_size), 
                        mesh_dim_names=("dp", "tp"))

# Step 1: TP
parallelize_module(model, mesh["tp"], tp_plan)

# Step 2: FSDP (在 TP 基础上再加 DP)
for layer in model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])

# 通信模式:
# Forward:  AllGather(DP mesh, NVLink) → TP compute (AllReduce)
# Backward: TP AllReduce → ReduceScatter(DP mesh)
```

## 6. 关键源码文件

| 文件 | 行数 | 核心职责 |
|------|------|---------|
| distributed/device_mesh.py | ~800 | DeviceMesh 类实现 |
| distributed/tensor/_api.py | ~600 | DTensor 核心 API |
| distributed/tensor/_dispatch.py | ~500 | DTensor 算子分派 |
| distributed/tensor/_redistribute.py | ~400 | Placement 转换通信 |
| distributed/tensor/parallel/ | ~1500 | TP plan 实现 |

## 7. 总结

```
DeviceMesh + DTensor 的价值:
┌──────────────────────────────────────────────────┐
│ 解决问题         │ 具体收益                        │
├──────────────────┼────────────────────────────────┤
│ 多维并行组合      │ 声明式网格, 无需手算 rank       │
│ 通信自动化        │ redistribute 自动推导通信类型   │
│ API 统一         │ FSDP + TP + PP 同一抽象         │
│ Checkpoint 兼容  │ DTensor state_dict 跨配置复用   │
│ torch.compile    │ DTensor 可追踪和优化            │
└──────────────────┴────────────────────────────────┘
```

## 8. DeviceMesh 内部实现

### 8.1 数据结构

```python
# torch/distributed/device_mesh.py
class DeviceMesh:
    device_type: str                    # "cuda", "cpu"
    mesh: torch.Tensor                  # N-D rank 布局 tensor
    mesh_dim_names: tuple[str, ...]     # 维度名称
    _flatten_mesh_list: list[int]       # 扁平化 rank 列表
    _thread_id: int                     # 创建线程 ID
    
    # 缓存的 ProcessGroup (惰性创建)
    _dim_group_infos: list[tuple[str, list[int]]]
    # 每个维度: (backend_name, group_ranks)
```

### 8.2 进程组创建策略

```
DeviceMesh ProcessGroup 创建策略:
─────────────────────────────────
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp","tp"))
  mesh_tensor = [[0,1,2,3],
                 [4,5,6,7]]

dim=0 (dp): 创建 4 个 group
  [0,4], [1,5], [2,6], [3,7]
  
dim=1 (tp): 创建 2 个 group
  [0,1,2,3], [4,5,6,7]

WHY 惰性创建?
  只有当 get_group(dim) 被调用时才创建 ProcessGroup
  避免初始化时创建所有可能的组 (节省 NCCL communicator 资源)
  
  如果 device_id 已设置: 使用 ncclCommSplit (快速)
  否则: ncclCommInitRank (慢, 需要 barrier)
```

### 8.3 子网格切片机制

```python
# mesh["tp"] 实现:
def __getitem__(self, mesh_dim_name: str) -> "DeviceMesh":
    """切出一维子网格"""
    dim = self.mesh_dim_names.index(mesh_dim_name)
    # 找到当前 rank 所在的切片
    submesh = self.mesh.select(other_dims, current_coord)
    return DeviceMesh(self.device_type, submesh, ...)

# 3D mesh 切 2D:
mesh_3d = init_device_mesh("cuda", (2,2,4), ("dp","pp","tp"))
mesh_2d = mesh_3d["dp", "tp"]  # 切出 dp×tp 子网格
```

## 9. DTensor 算子注册

### 9.1 Sharding Propagation

```
DTensor 算子分派流程:
────────────────────
用户调用: torch.matmul(X_dtensor, W_dtensor)
          │
          ▼
1. DTensor.__torch_dispatch__ 拦截
          │
          ▼
2. 查找 op 的 sharding rule (OpStrategy)
   - 每个 op 有注册的 placement 传播规则
   - matmul: [Shard(1)] @ [Shard(0)] → [Replicate]
          │
          ▼
3. 判断输入是否需要 redistribute
   - 如果当前 placement 不满足 rule → 插入通信
   - 例: X[Replicate] → X[Shard(0)] 需要 scatter
          │
          ▼
4. 在本地 tensor 上执行算子
   - 每个 rank 拿 local_tensor 调用 torch.matmul
          │
          ▼
5. 包装输出为 DTensor + 输出 placement
```

### 9.2 核心算子 Sharding Rules

| 算子 | 输入 Placement | 输出 Placement | 通信 |
|------|---------------|---------------|------|
| matmul(X,W) | [Shard(1)], [Shard(0)] | [Partial] | 无 (local mm) |
| matmul(X,W) | [Rep], [Shard(0)] | [Shard(1)] | 无 (local mm) |
| add(X,Y) | [Shard(0)], [Shard(0)] | [Shard(0)] | 无 (element-wise) |
| add(X,Y) | [Partial], [Rep] | [Partial] | 无 |
| softmax(X) | [Shard(0)] | [Shard(0)] | AllReduce (max,sum) |
| layernorm(X) | [Shard(0)] | [Shard(0)] | AllReduce (mean,var) |

### 9.3 Redistribute 通信映射

```
Placement 转换 → NCCL 操作:
────────────────────────────────────
Partial(SUM) → Replicate:    AllReduce(SUM)
Partial(SUM) → Shard(dim):   ReduceScatter
Shard(dim) → Replicate:      AllGather
Replicate → Shard(dim):      本地 chunk (无通信)
Shard(0) → Shard(1):         All-to-All

WHY 需要这些转换?
算子有特定的输入 placement 要求
如果实际 placement 不匹配, 必须先 redistribute
```

## 10. 与 Megatron parallel_state 对比

```
┌────────────────┬───────────────────┬────────────────────────┐
│ 维度           │ Megatron          │ DeviceMesh + DTensor    │
├────────────────┼───────────────────┼────────────────────────┤
│ 组管理         │ 全局变量 + getter  │ Mesh 对象 + dim 索引   │
│ 组创建         │ RankGenerator     │ init_device_mesh       │
│ Tensor 分片   │ 手动 scatter/gather│ DTensor + Placement    │
│ 通信插入       │ 显式 collective   │ 自动 redistribute      │
│ 代码复杂度     │ 高 (性能最优)     │ 低 (少量开销)           │
│ 灵活度         │ 固定 5D 并行      │ 任意维度组合            │
│ torch.compile │ 部分支持           │ 完全支持                │
│ 适用场景       │ 超大规模训练      │ 中等规模 + 快速开发     │
└────────────────┴───────────────────┴────────────────────────┘
```

## 11. 实际应用示例

```python
# 完整的 3D 并行 (DP + TP + PP) 使用 DeviceMesh:
# 64 GPU: dp=4, pp=4, tp=4

mesh = init_device_mesh("cuda", (4, 4, 4), 
                        mesh_dim_names=("dp", "pp", "tp"))

# PP: 手动按 stage 切分模型
stage_mesh = mesh["dp", "tp"]  # 每个 PP stage 的 2D mesh

# TP: 在每个 stage 内做 tensor parallel  
for layer in stage_layers:
    parallelize_module(layer, mesh["tp"], tp_plan)

# DP (FSDP): 在 TP 之上加数据并行
for layer in stage_layers:
    fully_shard(layer, mesh=mesh["dp"])

# 通信拓扑:
# TP: mesh["tp"] 组内 AllReduce (节点内 NVLink)
# DP: mesh["dp"] 组内 ReduceScatter (可能跨节点 IB)
# PP: mesh["pp"] 组内 Send/Recv (跨节点 IB)
```
