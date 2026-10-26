# FlagGems 深度源码分析 · 第三章：算子实现与自动 Codegen

> 知识组：`know-flaggems` ｜ 源码：FlagOpen/FlagGems 快照 `/tmp/flaggems_repo` @ `55a3ceda`
> 主题：`ops/` 一算子一文件的组织方式、`@pointwise_dynamic` 自动代码生成、`FunctionSchema`、
> `PointwiseDynamicFunction` 的按 rank 物化、`LibEntry` 快速分发，以及 codegen kernel 与手写 kernel 的工程对比。

## 3.1 总览：从 Python 调用到 Triton kernel 的五级流水线

```
用户代码                FlagGems 包装层              代码生成层                    Triton 层
──────────             ────────────────            ────────────────              ─────────────
torch.add(A,B,alpha=2)
  │  flag_gems.enable() 后，ATen 的 "aten::add" 被重定向到 FlagGems 实现
  ▼
ops/add.py: add(A, B, *, alpha=1)
  │  Python 分发梯子：complex? tensor×tensor? tensor×scalar?
  ▼
PointwiseDynamicFunction.__call__            (utils/pointwise_dynamic.py)
  │  prepare_args: 广播 shape、dtype 提升、分配输出 → 归纳出任务空间 rank ndim
  ▼
instantiate(ndim) ──── 首次调用时触发 codegen
  │  KernelGenerator   → 生成 Triton kernel 体（tile/bptr/GSL 策略）
  │  WrapperGenerator  → 生成 launcher（grid 计算、stride、launch）
  │  ModuleGenerator   → 拼装 module（imports + deps）→ 写入 ~/.flaggems 缓存
  ▼
LibEntry.run(*args)                          (utils/libentry.py)
  │  预计算 cache key → ConfigCache 命中则跳过 Triton 默认 dispatch
  ▼
生成的 @triton.jit kernel 在设备上执行（如 x + y * alpha）
```

**WHY（设计动机）**：PyTorch 一个 pointwise op 会被业务以任意 rank（0–8 维）、任意 dtype 组合、
任意 broadcasting 调用。为每个组合手写 kernel 是 O(rank × dtype^输入数) 的组合爆炸；FlagGems 把
"结构信息"（哪些参数是 tensor、输出怎么提升 dtype）声明进 `FunctionSchema`，把"数学"留在
`@triton.jit` 函数里，其余（tile 选择、grid、mask、指针计算、launcher）全部由 codegen 生成并按
ndim 缓存。热路径上只剩两次 Python 调用加一次字典查找。

## 3.2 ops/ 目录组织：一算子一文件

量化事实：`src/flag_gems/ops/` 下共 **910 个 `.py` 文件**（@55a3ceda 实测 `ls ops/*.py | wc -l`；
扣除包入口 `__init__.py` 即 909 个算子文件）。
文件名即 op 语义（`add.py`、`silu.py`、`rms_norm.py`、`softmax.py`、`_foreach_binary.py`），
多数文件内同时含 forward/backward/in-place 三个薄包装函数。

| 文件 | 覆盖的 ATen op | 实现模式 |
|---|---|---|
| ops/add.py | add / add_ | pointwise codegen（本章主角） |
| ops/silu.py | silu / silu_ / silu_backward | pointwise codegen |
| ops/softmax.py | _softmax | 手写 reduction kernel（@libentry） |
| ops/rms_norm.py | rms_norm + backward ×5 kernels | 手写 reduction（@libentry + autotune） |
| ops/_foreach_binary.py | _foreach_add/mul/sub… | foreach 批量循环 + 借用 codegen 数学 |

**为什么一算子一文件**：FlagGems 的注册入口是 `torch.library.Library("aten", "IMPL")`
（见 `src/flag_gems/__init__.py`，第一章详述）。注册表由 `_FULL_CONFIG` 驱动，op 名 ↔ 文件名
一一对应，使 include/exclude 过滤退化为字符串匹配；vendor 后端也得以"整文件替换"地覆盖单个算子。

## 3.3 FunctionSchema：把"结构"从"数学"里剥离出来

**问题类别**：这是经典的"接口描述语言（IDL）"问题——用声明式 schema 描述 op 的形状/类型
契约，标准方法就是 PyTorch 自家 `FunctionSchema`（`torch._C.FunctionSchema`）与 TVM
`TensorType` 的思路：结构可枚举 → 可校验、可推导、可生成。

- `utils/pointwise_dynamic.py:107`（原文：`class FunctionSchema:`）

FunctionSchema 只记录四类结构信息（构造函数 `pointwise_dynamic.py:118-185`）：

| 字段 | 含义 | add.py 示例 |
|---|---|---|
| `_is_tensor` | 每个参数是否为张量 | `[True, True, False]` |
| `_dtypes` | 非张量参数的静态类型（None=任意标量） | `[None, None, None]` |
| `_num_outputs` | 输出个数（输出不占输入位） | `1` |
| `_promotion_methods` | 每个输出的 dtype 提升规则 | `[(0, 1, DEFAULT)]` |

关键行为（逐字行）：

- `pointwise_dynamic.py:132`（原文：`        if promotion_methods is None:`）——promotion_methods
  必填，缺失直接 `raise ValueError(...)`（134 行），**没有隐式默认**：作者强制每个 op 显式声明
  dtype 语义，这是精度正确性的第一道闸门。
- `pointwise_dynamic.py:188`（原文：`    def canonicalize_promotion_methods(promotion_methods):`）
  ——把 `"DEFAULT"` 这类字符串映射为 `ELEMENTWISE_TYPE_PROMOTION_KIND` 枚举
  （`pointwise_dynamic.py:193`，原文：`                (*arg_indices, ELEMENTWISE_TYPE_PROMOTION_KIND[method])`），
  与 PyTorch `c10::dispatch::ELEMENTWISE_TYPE_PROMOTION_KIND` 一词对齐：DEFAULT=常用提升、
  UPSTAT=标量不提升、COMPLEX_TO_FLOAT=复数输出转实数。语义完全"借"自 PyTorch 官方
  type promotion 手册，保证与 ATen 逐位一致。
- `pointwise_dynamic.py:183`（原文：`        self._num_input_tensors = sum(self._is_tensor)`）——
  张量/标量计数在构造时一次算好；`_compute_input_id()`（245-256 行）生成
  "逻辑参数位 → 张量序号"映射，供后续指针命名（in0_ptr, in1_ptr…）。
- `pointwise_dynamic.py:242`（原文：`        sig = f'Pointwise: {", ".join(input_types)} -> {", ".join(output_types)}'`）——
  `signature()` 生成类似 `Pointwise: StridedBuffer, StridedBuffer, scalar -> StridedBuffer`
  的 JITScript 风格签名，用于缓存键与调试。

**WHY**：schema 是"编译期已知"的部分。把可以在装饰器时刻静态确定的一切（张量个数、提升
规则）从运行时剥离，后面的 prepare_args / codegen 才能免于逐次判断。

### 3.3.1 @pointwise_dynamic 装饰器：schema 的注册口

- `pointwise_dynamic.py:1930`（原文：`def pointwise_dynamic(`）——装饰器工厂。
- `pointwise_dynamic.py:1943`（原文：`            num_inputs = len(fn.arg_names)`）——
  当调用者只给 `promotion_methods` 不给 `is_tensor`（如 silu：单输入全张量）时，
  从 `@triton.jit` 函数的 `arg_names` 自动推参数个数——所以 silu.py 里的装饰器
  `@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])` 能省掉 `is_tensor=[True]`。
- `pointwise_dynamic.py:1951`（原文：`        return PointwiseDynamicFunction(op_desc, fn, config)`）——
  返回的不是函数，而是"可调用对象"，即 3.4 的分发主体。

### 3.3.2 dtype 推导伪代码（resolve/prepare_args 阶段）

```text
function PREPARE_ARGS(args, kwargs, schema):
  # ① out 张量回收：调用方传 out0=... 则复用（in-place 即 out0=A）
  outs := []; need_alloc := []
  for i in 0..num_output_tensors-1:
      if kwargs["out"+i] != None: outs.append(kwargs["out"+i])
      else: need_alloc.append(i)
  # ② dtype 提升：对每个待分配输出，取 promotion_methods[i] 声明的输入位
  for i in need_alloc:
      (idx..., method) := schema.promotion_methods[i]
      dtype := type_promotion(args[idx...], method)   # pointwise_dynamic.py:1652
  # ③ 任务空间 rank 推导：broadcast 所有输入 shape
  ndim := max( tensor.ndim for tensor in in_tensors )
  task_shape := broadcast(in_tensors[*].shape)        # 逐维 max
  # ④ >INT32_MAX 元素数时禁用 block pointer（防 64 位偏移开销）
  if tensors[0].numel() > INT32_MAX: config.prefer_block_pointer = False
  # ⑤ 按 (ndim, dtype 组合) 实例化/查缓存并启动
  return ndim, args, kwargs
```

**边界注记**：输入必须按位置传递（`prepare_args` 里 `check_tensor_attributes` 校验，
不符则 `raise ValueError`）；`out0=` 只认 schema 声明范围内的键，多余 kwargs 被丢弃。

## 3.4 代码生成：KernelGenerator / WrapperGenerator / ModuleGenerator 三层

`pointwise_dynamic.py` 的中段（265–1246 行）是三个生成器，职责切分：

| 生成器 | 行号 | 产出 | 关键决策 |
|---|---|---|---|
| `KernelGenerator` | `pointwise_dynamic.py:265` | Triton kernel 体 | tile 策略、block pointer、GSL 循环、0d 特例 |
| `WrapperGenerator` | `pointwise_dynamic.py:793` | Python launcher | grid 计算、task partition、shape 断言、输出返回 |
| `ModuleGenerator` | `pointwise_dynamic.py:1099` | 整个 `.py` 模块 | imports、依赖 `@triton.jit` 函数源码收集 |

### 3.4.1 按张量 rank 物化（instantiate / materialize）

- `PointwiseDynamicFunction.__init__`（`pointwise_dynamic.py:1316`）持有
  `self.overloads: Mapping[str, Callable] = {}`（1327 行）——**按 rank 缓存**生成的函数。
- `instantiate(ndim)`（`pointwise_dynamic.py:1792`，原文：`    def instantiate(self, ndim):`）：
  同一个 op 在 3D 输入与 5D 输入下是两个不同的生成函数；首次遇到某 rank 触发完整 codegen。
- `materialize(ndim)`（`pointwise_dynamic.py:1893`）返回 `PointwiseKernelMaterialization`
  （`pointwise_dynamic.py:1258`，原文：`class PointwiseKernelMaterialization:`）——把生成
  kernel 的结构句柄（`jit_function`、`max_tile_size`、`prefer_block_pointer`…）快照成
  **frozen dataclass**，专供 `torch.library.wrap_triton` 在 Dynamo 边界后使用（PT2 集成桥）。
- `KernelInfo`（`pointwise_dynamic.py:1248`，原文：`class KernelInfo:`）：
  `file_path / kernel_name / wrapper_name / ndim` 四元组，C++ 集成侧的定位符。

**为什么按 rank 而不是按 shape 物化**：shape 组合无穷，rank 只有 0–8；对同一 rank，
不同 shape 只改变 grid 与 mask 的运行时值，kernel 模板相同。这是生成函数数量收敛到
O(rank) 而非 O(shape) 的关键设计。

### 3.4.2 生成代码的落盘与缓存

生成产物写入 `~/.flaggems` 缓存目录（类 docstring，`pointwise_dynamic.py:1313`：
"The generated code are written out to the cache directory (defaults to ~/.flaggems)."），
依赖 `flag_gems/utils/code_cache.py` 的 `code_cache_dir()` 与 `code_utils.py` 的
`write_atomic`（`pointwise_dynamic.py:26-27` 原文：`from flag_gems.utils.code_cache import code_cache_dir` /
`from flag_gems.utils.code_utils import IndentedBuffer, write_atomic`）——原子写避免多进程
首跑互踩缓存文件。

### 3.4.3 WrapperGenerator 生成的 launcher 伪代码

```text
function GENERATED_WRAPPER(in0, in1, val0, out0=None):      # 形参由 schema 决定
  # 形状一致性断言（gen_same_shape_check, pointwise_dynamic.py:857）
  assert broadcastable(in0.shape, in1.shape)
  # 任务切分（gen_task_partition, pointwise_dynamic.py:865）
  ndim := in0.ndim
  grid  := balanced_partition(numel, TILE, MAX_GRID)   # _balanced_grid_partition(77行)
  # kernel 启动（gen_kernel_launch, pointwise_dynamic.py:950）
  kernel[grid](in0.ptr, in1.ptr, val0, out0.ptr, M, N, ..., TILE_N, num_warps=w)
  # 返回（gen_return, pointwise_dynamic.py:1070）
  return out0
```

## 3.5 复数路径与 in-place：codegen 之外的"半自动"层

codegen 只覆盖实数域：Triton 无 complex dtype。FlagGems 的策略是**实数域拆解 + 委托**：

- `ComplexMode`（`pointwise_dynamic.py:1288`，原文：`class ComplexMode(Enum):`）：
  `NONE / ELEMENTWISE / CROSS` 三态，docstring 写明两种策略——1290 行：
  `    ELEMENTWISE = auto()  # add/sub: view_as_real → same kernel → view_as_complex`；
  1291 行：`    CROSS = auto()  # mul/div: split ar/ai/br/bi → cross_kernel`。
- `register_complex`（`pointwise_dynamic.py:1357`，原文：`    def register_complex(`）
  把 `ComplexStrategy(mode, cross_kernel, tensorize_scalars, fallback_target)`（1295 行）
  挂到函数对象上。
- 调用入口（`pointwise_dynamic.py:1380`，原文：`    def __call__(self, *args, **kwargs):`）：
  `_should_use_complex_path`（1459 行）→ 复数走 `_call_complex_dispatch`，实数走 `_call_real_impl`。
- 标量变体的复数委托（`ops/add.py:50-52`）：`tensorize_scalars=True` 时把 Python 标量
  升格为 0-d 张量后整体转发 `fallback_target=add_func`——T+T 版本有全功能复数实现，
  避免"每个变体 × 每种复数组合"再 codegen 一遍。

`add()` 的复数分支（`ops/add.py:66-103`）演示 ELEMENTWISE 三段式：
`Ar = torch.view_as_real(A)`（68 行）→ `add_func(Ar, Br, alpha)`（72 行）→
`torch.view_as_complex(out_real)`（73 行）。**边界注记**：`ops/add.py:80` 的注释解释了
为什么实标量要先构造实视图再 broadcast——原文：
`                # creating a complex-dtype tensor (unsupported by Triton kernels)`，
即 Triton kernel 不能接收 complex dtype 的张量。

in-place 是另一条半自动路径：`ops/add.py:121`（原文：`        return add_func(A, B, alpha, out0=A)`）
让 codegen 的输出缓冲直接指向输入 A，省一次分配，但要求 A 可写。

## 3.6 LibEntry：绕过 Triton 默认 dispatch 的快速通道

**问题类别**：解释器开销问题。Triton 官方 `JITFunction.run` 每次调用都要做完整的
arg binding、specialization、多级 cache 查找（Python 层微秒级 × 每 step 数千次 kernel
启动 = 训练可感知开销）。标准方法是**预计算 + 单级字典缓存**——这正是 `LibEntry`。

- `utils/libentry.py:1369`（原文：`class LibEntry(triton.KernelInterface):`）
- `utils/libentry.py:1703`（原文：`def libentry():`）→ `decorator` 返回 `LibEntry(fn)`
  （1706 行 `def decorator(fn):` 内 1707 行 `return LibEntry(fn)`），与 `@triton.jit` 叠加使用：`@libentry()` 在最外层
  （softmax.py:30、rms_norm.py:34）。

### 3.6.1 三类参数 → 一个 entry_key

`LibEntry.__init__`（1370-1401 行）把 JIT 参数预分为三组：

- `libentry.py:1384`（原文：`        self.specialize_indices = [`）——specialize 参数
  （非 constexpr 且未标 `do_not_specialize`）：指针按 `data_ptr() % divisibility` 判对齐
  （1452 行，`self.divisibility = 16` 初始化于 1376 行），整数按是否为 1 特化；
- `libentry.py:1389`（原文：`        self.do_not_specialize_indices = [`）——标了
  `do_not_specialize` 的参数（如 rms_norm 的 `eps`）只进 key，不触发重编译；
- constexpr 参数直接进 key。

`key()`（`libentry.py:1434`）把三组打包成 tuple；`run()`（`libentry.py:1472`）主循环：

- `libentry.py:1529`（原文：`        while entry_key not in cache:`）——单级缓存主循环：
  命中则跳过 Triton 全部运行时逻辑；未命中才进加锁慢路径。
- `libentry.py:1535`（原文：`                kernel = self.fn.run(*args, **kwargs)`）——
  慢路径调用真正的 Triton `JITFunction.run`（Autotuner/Heuristics 已在装饰器链上生效）。
- `libentry.py:1574`（原文：`                cache[entry_key] = (`）——把
  `(kernel, constexprs, tune_constexprs, heur_constexprs, launch_pre_hooks)` 五元组缓存；
  Autotuner 的 `best_config`（1544 行）与 Heuristics 的值（1554-1563 行）在首跑后都被
  "榨干"成具体值存起来。
- `libentry.py:1591`（原文：`        if callable(grid):`）——grid 闭包用 meta 字典求值一次。
- `libentry.py:1626`（原文：`            kernel[grid[0:3]](*k_args.values())`）——
  直接用编译产物启动 kernel，绕过 Triton dispatcher。

**边界注记**：① 首 run 全局加锁（1529-1532 行 `with self.lock`，multiprocessing.Lock），
Triton runtime 非线程安全——多进程训练下首个 batch 有一次串行化；② 每设备独立缓存
`kernel_cache = tuple(dict() for _ in range(DEVICE_COUNT))`（1377 行）；③ CPU 走独立
`_cpu_cache`（1525-1528 行）；④ macOS 用 threading.Lock（1394-1398 行，文件描述符限制）；
⑤ FlagTune 调优器挂钩（`_apply_flagtune`，1419-1432 行）在配置变化时清缓存重调。

### 3.6.2 调用路径对比（热路径开销模型）

```
原生 Triton:  python call → JITFunction.run
              → binder/specialization 检查 → cache 查找(多级) → grid 求值 → launch
LibEntry:     python call → LibEntry.run
              → 三组参数扫描(O(参数数)) → dict[entry_key] 命中 → grid 求值 → launch
```

LibEntry 砍掉的是 arg binding 与 specialization 的**重复**计算——对每 step 调用上千次的
optimizer/loss 算子收益最大；这正是 `@libentry()` 只用于手写高频 kernel（softmax/rms_norm
等 reduction 类）而非 codegen kernel 的原因：codegen kernel 的调用方（PointwiseDynamicFunction）
自己已经持有按 rank 缓存的 overload。

## 3.7 代表算子四象限：codegen 与手写的分工

### 3.7.1 softmax（手写 reduction，408 行）

softmax 有跨行依赖（max/sum 归约），不属于 pointwise，FlagGems 用**手写 Triton kernel +
@libentry + heuristics** 实现：

- `ops/softmax.py:33`（原文：`def softmax_kernel_non_inner(`）——非 innermost 维归约
  kernel；`ops/softmax.py:30`（原文：`@libentry()`）挂快速分发。
- `ops/softmax.py:31`（原文：`@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))`）
  ——TILE_N/TILE_K/ONE_TILE_PER_CTA 由 runtime 启发式决定（第 38-56 行 kernel 内
  `if ONE_TILE_PER_CTA:` 双分支：小行一次 CTA 装下；否则 online-softmax 两遍循环，
  53 行注释原文：`        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject`）。
- **WHY fp32 归约**：fp16/bf16 直接 `tl.exp` 在部分 vendor 后端非法，且 fp32 累加精度更高——
  这类决策无法从 op 语义推导，必须人写。

### 3.7.2 rms_norm（手写 + autotune，405 行）

- `ops/rms_norm.py:404`（原文：`def rms_norm(x, normalized_shape, weight, eps=1e-5):`）
  是薄分发层，下面挂 5 个 kernel：`rms_norm_kernel`（36 行）、`rms_norm_loop_kernel`
  （82 行，配 `@triton.autotune` 77 行）、backward 的 dx/dw 各两版本。
- 全部 kernel 用 `@triton.jit(do_not_specialize=["eps"])`（35 行）——eps 不参与特化，
  避免每个 eps 值重编译；`@libentry()` 缓存其分发。norm 类是"手写 + autotune"的典型：
  归约维度（normalized_shape）与 N 的取值空间大，值得 runtime 调优。

### 3.7.3 foreach 系（复用 codegen 的数学，1157 行两个文件）

foreach 系列是 **codegen 与手写的中间形态**：单张量数学来自 pointwise codegen 体系，
批量执行自己写。

- `ops/_foreach_unary.py:92`（原文：`def _scalar_fn(pointwise_fn) -> Callable:`）——
  通过 `pointwise_fn._scalar_fn` 把已注册的 pointwise 算子内部的 `@triton.jit` 标量函数
  **整体借出来**（93-99 行 docstring："the foreach executor needs exactly the same object
  as a ``tl.constexpr`` callable"），30+ 个一元 foreach op 由此复用既有数学而不重述。
- `ops/_foreach_binary.py:52`（原文：`# takes an ``alpha`` argument bound by ``pointwise_dynamic``, and several of`）
  ——**边界注记**：为什么不直接调单张量算子？原文注释（52-55 行）给出两个原因：
  add_func 带 pointwise_dynamic 绑定的 alpha 参数；部分单张量算子内部 cast float32，
  会悄悄改变整数 foreach 的结果 dtype。所以 foreach 用裸 `_add_fn/_sub_fn/_mul_fn/_div_fn`
  （58/69/74/84 行）。

### 3.7.4 silu（纯 codegen，59 行）

- `ops/silu.py:44`（原文：`def silu(self):`）——全部业务逻辑 4 行：
  `silu_forward(self)` + 日志。`ops/silu.py:26`（原文：`@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])`）
  ——单输入全张量，最简 schema；`ops/silu.py:29`（原文：`    x_fp32 = x.to(tl.float32)`）
  ——数值稳定性在标量函数内部处理（先转 fp32 再除）。
- **边界注记**：silu_ 原地版本直接 `silu_forward(A, out0=A)`（56-58 行），与 add_ 同一机制。

### 3.7.5 四象限分工总结

| 象限 | 特征 | 机制 | 例 |
|---|---|---|---|
| 纯 pointwise | 无归约、逐元素 | @pointwise_dynamic 全自动 codegen | add/silu/mul |
| 带归约 | 跨行/跨列依赖 | 手写 kernel + @libentry(+autotune/heuristics) | softmax/rms_norm |
| 批量 foreach | N 个张量同运算 | 借用 _scalar_fn + 手写批量 executor | _foreach_* |
| 复数/特殊 | Triton 不支持的 dtype | Python 层拆解 + 委托 codegen | add(complex)/in-place |

**WHY 这个分工**：codegen 的比较优势在"结构规则、数学简单、调用频度均匀"的算子——
一份 schema + 一行数学换全 rank/全 dtype 覆盖；一旦出现跨元素依赖（reduction）、
不规则内存访问或 vendor 特异调优需求，生成模板的表达成本超过手写成本，就回到手写路线。

## 3.8 Codegen vs 手写 kernel：工程对比

| 维度 | @pointwise_dynamic codegen | 手写 kernel（@libentry） |
|---|---|---|
| 代码量 | add: 127 行覆盖 T+T/T+S/S+T×complex×inplace | softmax: 408 行只覆盖 _softmax 一个 op |
| 新算子成本 | 1 行装饰器 + 1 行数学 | 全套 kernel + launcher + 调优 |
| 覆盖面 | 自动覆盖 0-8 维、任意 dtype 组合 | 只覆盖作者写过的 case |
| dtype 提升 | schema 声明，与 ATen 官方规则对齐（canonicalize） | 手工保证，易漏边界 |
| 性能上限 | 通用模板：tile/bptr/GSL 启发式选择 | 可针对算子手工调（autotune、双 pass、online softmax） |
| 调试透明度 | 生成代码落盘 ~/.flaggems 可读 | 源码即真相 |
| 归约支持 | 无（pointwise only） | 原生支持 |
| PT2/编译器集成 | materialization 冻结句柄（frozen dataclass） | 需自行暴露 jit_function |
| 典型文件规模 | silu 59 行 / add 127 行 | softmax 408 行 / rms_norm 405 行 |

**结论**：FlagGems 用 codegen 吞掉"长尾 pointwise"（数量大、单个简单、覆盖要求高），
把手写预算集中在"高频 reduction + vendor 调优"上；两层共享 LibEntry/LibTuner 基础设施
（第一章 2.4/2.6 节），形成完整的算子金字塔。

## 3.9 证据锚点汇总

| # | 锚点 | 逐字引用行 |
|---|---|---|
| 1 | `ops/add.py:26` | `@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])` |
| 2 | `ops/add.py:29` | `    return x + y * alpha` |
| 3 | `ops/add.py:58` | `def add(A, B, *, alpha=1):` |
| 4 | `ops/add.py:49` | `add_func.register_complex(mode=ComplexMode.ELEMENTWISE)` |
| 5 | `pointwise_dynamic.py:107` | `class FunctionSchema:` |
| 6 | `pointwise_dynamic.py:1310` | `class PointwiseDynamicFunction:` |
| 7 | `pointwise_dynamic.py:1380` | `    def __call__(self, *args, **kwargs):` |
| 8 | `pointwise_dynamic.py:1357` | `    def register_complex(` |
| 9 | `pointwise_dynamic.py:1930` | `def pointwise_dynamic(` |
| 10 | `pointwise_dynamic.py:1943` | `            num_inputs = len(fn.arg_names)` |
| 11 | `pointwise_dynamic.py:1258` | `class PointwiseKernelMaterialization:` |
| 12 | `pointwise_dynamic.py:1288` | `class ComplexMode(Enum):` |
| 13 | `libentry.py:1369` | `class LibEntry(triton.KernelInterface):` |
| 14 | `libentry.py:1529` | `        while entry_key not in cache:` |
| 15 | `libentry.py:1626` | `            kernel[grid[0:3]](*k_args.values())` |
| 16 | `ops/softmax.py:33` | `def softmax_kernel_non_inner(` |
| 17 | `ops/rms_norm.py:404` | `def rms_norm(x, normalized_shape, weight, eps=1e-5):` |
| 18 | `ops/_foreach_unary.py:92` | `def _scalar_fn(pointwise_fn) -> Callable:` |
| 19 | `ops/silu.py:44` | `def silu(self):` |

以上 19 个锚点均以 `grep -n` / `sed -n` 逐字核对于 /tmp/flaggems_repo @55a3ceda。
