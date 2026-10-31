# prompt 不变式契约（v64 起）

本文件取代 23 个 `tests/test_prompt_*.py` 的**逐字断言**。原因不是"测试碍事"，而是这些
测试编码的是**上一版的具体措辞**，不是**被防住的失败模式**——一次彻底重写必然让它们全红，
而为了让它们变绿只能把新 prompt 反向锁死在旧句子上。这正是 prompt 只增不减的棘轮
（prop_62a73614 记录过，v64 实测确认：131 个红测试）。

## 一、被退役的测试族（逐字断言型）

| 测试文件 | 断言形态 | 退役理由 |
|---|---|---|
| test_prompt_scientific_protocol.py | 逐字长句 | 编码 2.x 具体措辞 |
| test_prompt_static_clause_anti_decay.py | 逐字长句 | 同上 |
| test_prompt_plain_language.py | 逐字句子 | 同上 |
| test_prompt_definition_axis_clause.py | 逐字长句（7 处） | 同上 |
| test_prompt_clause_anti_decay.py | clause 必须以固定前缀开头 | 绑定旧 bullet 结构 |
| test_prompt_source_independence_clause.py | `index("2.1a A CHECK IS ONLY AS GOOD")` | 绑定旧节名 |
| test_prompt_semantic_cosourcing_clause.py | `index("2.1b SEMANTIC CO-SOURCING")` | 绑定旧节名 |
| test_prompt_reading_ledger.py | 逐字长句 | 编码 2.9 措辞 |
| test_prompt_reading_value_clause.py | 逐字长句 | 同上 |
| test_prompt_action_requirement_clause.py | `find("AN ACTION THE TASK NAMES ...")` | 绑定旧标题 |
| test_prompt_g1c_waiting_not_work.py | 逐字 | 编码 G1c 措辞 |
| test_prompt_g3_adjudication.py | 逐字 | 编码 G3 措辞 |
| test_prompt_g3_build_toward_bound.py | 逐字 | 同上 |
| test_prompt_deliberation_budget.py | 逐字 | 编码 TIME BUDGET 措辞 |
| test_prompt_budget_checkpoint.py | 逐字 + 要求 bullet 行 | 绑定旧 bullet 结构 |
| test_prompt_acceptance_facet_manifest.py | 固定前缀 "Acceptance-facet manifest" | 绑定旧 bullet 名 |
| test_prompt_stated_value_oracle.py | 逐字 | 编码 2.3b 措辞 |
| test_prompt_rsi_step.py | 逐字 | 编码 RSI 模板措辞 |
| test_prompt_wait_deadline.py | 逐字 | 编码等待条款措辞 |
| test_context_v3.py（`TestPromptIntegrity` 部分） | 逐字 + 要求含中文示例 | 绑定旧措辞与旧语言策略 |
| test_guards.py（`test_frame_step_*` 4 项） | 要求 FRAME 含 "brute"/"documented"/"write-through"/"reference artifacts" | 绑定旧措辞 |
| test_plan_premortem.py（2 项） | 要求 "if reference artifacts are present" | 同上 |
| test_guard_minimal_legal_forms.py（2 项） | 要求 prompt 内含转义后的 `convention={{"name": "none"}}` | 编码旧 STAGE 列表 |

**保留不改**：`test_prompt_builder.py`（注入链路）、`test_prompt_watchdog.py`（超时）、
以及全部非 prompt 的代码测试。

## 二、v64 的替代契约（语义级，可执行）

这些是新 prompt **必须**满足的不变式。它们判"有没有这一节、这一节是否可执行"，
不判具体句子。实现见 `tests/test_prompt_v64_contract.py`（待建，一条断言一句）。

| # | 不变式 | 判定 |
|---|---|---|
| C1 | 名称不变 | 首行含 `FlagScale Agent` |
| C2 | 四个占位符齐全且 `.format()` 可渲染 | `{cwd} {tools} {skills} {knowledge}` |
| C3 | 总规模受控 | `len(SYSTEM_PROMPT_STATIC) <= 15000` |
| C4 | 十条纪律各有一节 | 节标题数 == 10 |
| C5 | 四条通用纪律都在 | 文本含 falsifiability / systematicity / error cost / verification≠validation 的语义 |
| C6 | 领域表存在且覆盖 6 个领域 | 含 math / physics / biology / CS / statistics / engineering |
| C7 | 三个数据日期问 | 存在"谁产出/换算法/会不会红"三问 |
| C8 | 收尾四步 | 存在 9.1–9.4 |
| C9 | 回复格式三渠道 | 含 Language / Task result / RSI result |
| C10 | 两个 marker 都在 | `[TASK_COMPLETE]` 与 `[NEED_USER_INPUT]` |
| C11 | 无旧节名残留（防回流） | 不含 `PRINCIPLE 2` / `G1 BYTES, NOT MODEL` / `2.1b SEMANTIC` |
| C12 | 无中文散文（作者留言除外） | 汉字只出现在作者留言段 |

## 三、为什么这不损失防护力

被退役的测试共覆盖约 20 类失败模式（自证检查、定义轴、阅读台账、预算检查点、
stated value、facet manifest…）。这些模式的**实质**都已进入 v64 prompt 的第 1–9 节
（映射见 `analysis/prompt_v64_refactor_plan.md` 第 2、4 节）。

退役的是"用哪句话说的"，保留的是"这件事有没有被要求"。若某个失败模式在 v64 里
真的缺席，C1–C12 抓不到——所以还有一条**外部**验证：4 个高方差任务的 A/B 对照。
