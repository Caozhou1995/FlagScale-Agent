# Retired 2026-11-01 — plan 字段化繁为简（用户裁决）

被删机制：oracle(5键)/oracle_step_id/calibration(4键)/requirements_read(3键)/convention/
deliverable_reading/assumptions/acceptance_act/differential/negative_control/
bound_margin/arbiter_act/oracle_act + guard/verification.py + react/oracle.py + react/instruments.py

这些测试的被测对象已不存在。恢复机制时从 git 历史找回。

## 2026-11-01 补充退役（v65 收口时发现）
- test_convention_discrimination_tool.py, test_g4_branch_coverage.py：断言 v64 重写已删除的 prompt 子句（2.x 措辞族，`assert ... in PROMPT`）。v64 用户指令：单测按重构后思想处理，不再保留旧措辞绑定。恢复=git checkout HEAD~1 -- tests/<file>。
