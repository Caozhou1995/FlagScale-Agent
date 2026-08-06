# FlagScale Repo 结构与架构 深度源码分析

## 1. 概述与设计动机

FlagScale 是一个统一的大模型训练/推理/部署框架，通过 Hydra 配置体系和 Runner 架构将不同任务类型（train/inference/serve/compress/rl）统一在一个入口下。

**核心设计思想**：一份 YAML 配置驱动从环境准备到任务执行的全链路，通过 Backend + Launcher 的组合解耦"做什么"和"怎么调度"。

**解决的问题**：
- 大模型训练/推理/部署各自为政，启动方式不统一
- 多节点调度逻辑（SSH/Cloud）与业务逻辑耦合
- 配置散落在脚本和环境变量中，难以版本化和复现

## 2. 源码定位

| 组件 | 路径 | 行数 | 职责 |
|------|------|------|------|
| 包入口 | `flagscale/__init__.py` | 42 | 版本获取 |
| Runner 基类（新） | `flagscale/runner/runner_base.py` | 92 | 新架构：Backend+Launcher 组合 |
| Runner 基类（旧） | `flagscale/runner/runner_base_legacy.py` | 37 | 旧架构：继承式 Runner |
| Runner 工厂 | `flagscale/runner/runner_factory.py` | 79 | 注册表模式分发 Backend/Launcher |
| 训练 Runner | `flagscale/runner/runner_train.py` | 928 | SSHTrainRunner + 脚本生成 |
| 推理 Runner | `flagscale/runner/runner_inference.py` | 322 | 推理启动逻辑 |
| 部署 Runner | `flagscale/runner/runner_serve.py` | 1441 | 多引擎部署（vLLM/SGLang/原生） |
| 工具函数 | `flagscale/runner/utils.py` | 1210 | hostfile 解析、目录管理、命令执行 |
| 训练配置 | `flagscale/train/train_config.py` | 340 | Megatron 训练配置补充与校验 |
| CLI 入口 | `flagscale/runner/__init__.py` | 13 | `run`/`stop`/`query` CLI 命令注册 |

## 3. 架构总览

### 3.1 顶层目录结构

```
FlagScale/
├── flagscale/              # 核心 Python 包
│   ├── __init__.py         # 版本管理
│   ├── runner/             # 任务调度层（Backend + Launcher）
│   ├── train/              # 训练入口脚本与配置逻辑
│   ├── inference/          # 推理入口脚本
│   ├── serve/              # 部署引擎与路由
│   ├── compress/           # 压缩/量化
│   ├── models/             # 模型注册与适配
│   ├── platforms/          # 硬件平台适配
│   ├── transformations/    # 模型变换 hook
│   ├── eval/               # 评测
│   └── agent/              # Agent 集成
├── examples/               # 各模型的完整配置示例
│   ├── qwen3/conf/         # Qwen3 各规模配置
│   ├── deepseek_v3/conf/   # DeepSeek-V3 配置
│   ├── llama3/conf/        # LLaMA3 配置
│   └── ...                 # 30+ 模型
├── tools/                  # 数据处理、checkpoint 转换等工具
├── tests/                  # 单元测试 + 集成测试
├── requirements/           # 依赖分组（cuda/train, cuda/serve 等）
├── docker/                 # 各平台 Docker 构建
└── docs/                   # 文档
```

### 3.2 Runner 核心类关系图

```
                    ┌─────────────────────────────────┐
                    │         CLI Entry Point          │
                    │  flagscale run/stop/query ...    │
                    └────────────────┬────────────────┘
                                     │ Hydra compose config
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Runner (runner_base.py)      │
                    │  - 解析 task_type + backend      │
                    │  - RunnerFactory 获取组件        │
                    │  - self.backend + self.launcher  │
                    └────────────┬────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
     ┌────────────┐    ┌──────────────┐    ┌──────────────┐
     │  Backend   │    │   Launcher   │    │  旧架构      │
     │ (做什么)   │    │  (怎么调度)  │    │ RunnerBase   │
     ├────────────┤    ├──────────────┤    │  Legacy      │
     │ megatron   │    │ ssh          │    └──────────────┘
     │ vllm       │    │ cloud        │         ↑
     │ native_*   │    │ local        │    SSHTrainRunner
     │ verl       │    └──────────────┘    (runner_train.py)
     └────────────┘
```

**设计决策**：新旧架构并存。训练任务仍用旧的 `SSHTrainRunner`（继承 `RunnerBase`），推理/部署已迁移到新的 Backend+Launcher 组合模式。

### 3.3 任务类型与后端映射

来源：`runner_base.py:24-30`

```python
TASK_TO_BACKEND_MAP = {
    "train":     ["megatron", "native"],      # 训练：Megatron 或原生
    "inference": ["vllm"],                    # 推理：vLLM
    "compress":  ["native", None],            # 压缩：原生
    "serve":     ["vllm", "sglang", "llama_cpp", "native", None],  # 部署：多引擎
    "rl":        ["verl"],                    # 强化学习：veRL
}
```

## 4. 核心模块分析

### 4.1 Runner 基类（新架构）

**源码**：`flagscale/runner/runner_base.py:33-92`

**设计动机**：将"业务逻辑"（Backend：如何构建命令参数）和"调度方式"（Launcher：如何分发到多节点）解耦，使新增 backend 或 launcher 只需注册一个类。

**初始化流程**：
```
Runner.__init__(config)
  1. parse_hostfile → self.resources          (L36-37)
  2. 确定 task_type                           (L38)
  3. 确定 backend_type (含 native 归一化)     (L42-71)
  4. 验证 task_type + backend 合法性          (L74-78)
  5. RunnerFactory.get_backend(type)(config)   (L80)
  6. RunnerFactory.get_launcher(type)(config, backend)  (L81)
```

**边界条件**：
- `train`/`inference`/`rl` 必须显式指定 backend（L51-55）
- `compress`/`serve` backend 可选，默认 "native"（L57-59）
- `native` 会被归一化为 `native_{task_type}`（L62-63）
- `serve` 且 `launcher=cloud` 时强制 backend=vllm（L44-45）

### 4.2 Runner 工厂

**源码**：`flagscale/runner/runner_factory.py`（79 行）

**设计动机**：注册表模式，避免 import 耦合。Backend 和 Launcher 通过装饰器自注册。

```python
# 注册模式（伪代码，runner_factory.py:L20-50）
class RunnerFactory:
    _backend_registry = {}
    _launcher_registry = {}

    @classmethod
    def register_backend(cls, name):
        def decorator(backend_cls):
            cls._backend_registry[name] = backend_cls
            return backend_cls
        return decorator

    @classmethod
    def get_backend(cls, name):
        return cls._backend_registry[name]
```

### 4.3 训练 Runner（旧架构）

**源码**：`flagscale/runner/runner_train.py`（928 行）

**设计动机**：训练涉及复杂的多节点协调（torchrun、hostfile、监控），还没迁移到新的 Backend+Launcher 模式。仍继承 `RunnerBase`（旧）。

**关键函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `_get_args_megatron` | L50-68 | 将 train.{system,model,data} 展平为 `--key value` 命令行参数 |
| `_get_args_native` | L71-90 | 原生训练：传 Hydra config.yaml 路径 |
| `_update_config_train` | L93-163 | 补全 checkpoint/logging/tokenizer 路径 |
| `_get_runner_cmd_train` | L166-230 | 构建 torchrun 命令参数 |
| `_generate_run_script_train` | L233-340 | 生成 bash 运行脚本 |
| `SSHTrainRunner._run_each` | L421-500 | 单节点执行逻辑 |
| `SSHTrainRunner.run` | L502-596 | 多节点并发调度 |
| `SSHTrainRunner.stop` | L598-633 | 停止所有节点 |

### 4.4 工具函数库

**源码**：`flagscale/runner/utils.py`（1210 行）

关键函数清单：

| 函数 | 行号 | 职责 |
|------|------|------|
| `setup_exp_dir` | L75-83 | 解析并创建实验目录 |
| `setup_logging_dirs` | L111-148 | 创建日志目录结构（scripts/pids） |
| `flatten_dict_to_args` | L489-510 | 递归展平 dict 为 `--key value` 列表 |
| `parse_hostfile` | L540+ | 解析 hostfile 获取节点资源 |
| `get_nnodes` / `get_nproc_per_node` | L511+ | 节点/进程数推断 |
| `resolve_path` | 早期 | 路径解析（支持相对路径/环境变量） |
| `run_ssh_command` / `run_scp_command` | 中后部 | SSH 远程执行与文件传输 |
| `find_latest_stdout_log` | 后部 | 查找最新的 stdout.log 文件 |

## 5. 设计决策对比表

| 维度 | 新架构（Backend+Launcher） | 旧架构（SSHTrainRunner） | 选择理由 |
|------|---------------------------|-------------------------|----------|
| 解耦程度 | 高：Backend 与 Launcher 独立 | 低：调度逻辑内嵌 | 新场景（cloud）需要 |
| 当前使用 | inference、serve、rl | train | 训练逻辑复杂，迁移未完成 |
| 扩展方式 | 装饰器注册 | 继承重写 | 注册模式更灵活 |
| 配置传递 | 统一 DictConfig | 统一 DictConfig | 相同 |
| 脚本生成 | Launcher 负责 | Runner 内部函数 | 旧架构脚本生成与调度耦合 |

## 6. 边界条件与约束

1. **训练仍强依赖旧架构**：`SSHTrainRunner` 直接继承 `RunnerBase`（legacy），不经过 `RunnerFactory`
2. **backend=native 的含义**：自定义训练入口，不走 Megatron，但仍用 torchrun 调度
3. **hostfile 格式**：`<ip> slots=<N> [type=<device>]`，type 可选（用于异构训练）
4. **no_shared_fs 模式**：日志目录结构不含 host IP（统一为 `host/`），脚本需 SCP 传送
5. **per_node_task 模式**：每节点独立任务（nnodes=1, node_rank=0），用于 data preprocessing

## 7. 配置建议

- 新增训练模型：在 `examples/{model}/conf/` 下按两级结构组织配置
- 新增任务类型：在 `TASK_TO_BACKEND_MAP` 注册 + 实现对应 Backend 类
- 新增调度方式：通过 `@RunnerFactory.register_launcher` 注册
- 调试训练配置：使用 `flagscale run ... --dryrun` 只生成脚本不执行
