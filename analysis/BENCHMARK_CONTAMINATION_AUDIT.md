# Benchmark 污染审计：四层检查清单

适用：任何"agent 在 benchmark 上分数提升"的结论。**先证明环境未被污染，再讨论能力。**
本清单来自 2026-10-20 tbench 4 任务集的事故（详见 `analysis/CLEANUP_REPORT.md`）。

## 判据（先跑这个，成本最低）

按提交二分 `prompt.py` 的泄漏词命中数，与 reward 表做时序对齐：

```bash
cd <repo>
for c in $(git log --format=%h --reverse -- flagscale_agent/react/prompt.py | tail -25); do
  d=$(git show -s --format=%ci $c | cut -c6-16)
  n=$(git show $c:flagscale_agent/react/prompt.py 2>/dev/null |
      grep -ciE "mteb|torchmetrics|sentence-transformers|primers?\.fasta|oligotm|melting temperature|bijection|annealed|stolen_A1|cli_tool")
  echo "$c $d hits=$n"
done
```

若 **3/4 与 4/4 全部落在 hits>0 的提交之后**，而 hits=0 时代最高只有 2/4，则提升归因于泄漏。

第二个判据——分布突变：某任务同一 HEAD 前后从 `0/5 -> 11/11`、`0/3 -> 5/5`。
能力提升不会从 0 直跳 100%；改的是一条绕过能力的通路。

## 四层通道（逐层都要在**容器内**验证，不是宿主机）

### 层1 静态 prompt
agent 每轮必读。若出现任务专有名词（具体库名、工件名、生物学术语、判分阈值），即为泄漏。
越界线：**任何任务身份信息**（库名、文件名、任务名词、精确数值）。方法论保留，身份全部去掉。
验证：`grep -ciE "<leak words>" flagscale_agent/react/prompt.py` → 必须 0。

### 层2 运行时 nudge（最隐蔽）
drafted 时以"教学示例"包装，但它在 agent 的消息流里。检查
`oracle.py` / `judge.py` / `verification.py` 里所有**会进入 agent 消息**的字符串。
本案实例：`oracle.py:105-106` 逐字含「gave 56 nt / bound 15-45 / (15-17 nt)」、
`judge.py:754` 与 `verification.py:1248` 含「Tm within 5 degrees」。

验证：在**该轮自己的容器**里
```bash
docker exec <trial-container> grep -rn "<leak words>" /opt/fsagent-src/flagscale_agent/ | wc -l   # 必须 0
```

### 层3 仓库 bind-mount
整个仓库挂进容器时，`tests/` 里的判分器副本、`analysis/` 里的求解器、`examples/` 都可见。
**实测才算数**：
```bash
docker exec <c> ls <repo>/tests/test_constraint_coverage_gate.py   # 旧：存在（逐字含阈值）
docker exec <c> ls /opt/fsagent-src/tests                          # 新：No such file or directory
docker exec <c> ls <old-repo-path>                                 # 新：No such file or directory
```
修法：只挂 staging 树（仅可导入包 + pyproject/README/LICENSE），并带硬断言禁止 `tests/` `analysis/`
`examples/` `build/` `*.egg-info`。注意 `pyproject.toml` 的 `readme = "README.md"` 是必需项，
所以不能只挂子目录。

### 层4 任务级注入
把官方 `solve.sh` 实现、隐藏判分器构造写进 instruction 尾部并强制「RUN it, by ABSOLUTE path」。
这比层1-3 更严重：不给选择权，等于让 agent 先抄再验。**整体移除**，不做"措辞弱化"。

## 两个反复踩的陷阱

1. **grep=0 在"目录不存在"时也成立。** 必须同时断言目录存在且关键文件 md5/字节数与 HEAD 一致：
   ```bash
   docker exec <c> bash -c 'ls -d /opt/fsagent-src && md5sum /opt/fsagent-src/flagscale_agent/react/oracle.py'
   md5sum flagscale_agent/react/oracle.py   # 两侧必须相同
   ```
2. **容器在 setup 时快照复制 staging。**「先启动 trial、后修 staging」的容器仍带旧快照。
   判断某轮是否干净，**必须进该轮容器** grep。本案 `four_m42_clean` 就因此在容器内仍留 3 处层2泄漏。

## 附带必查：测试本身固化了泄漏

清理后出现 4 failed，原因是测试断言泄漏词**必须在**（`assert "get_model_meta" in PROMPT` 等）。
这些测试是泄漏的形态。改法：断言「规则在、身份不在」，并加反向断言（泄漏词不得回来）。

## 收尾

- 回退前先做 `git bundle` 快照（`git bundle verify` 应报 "records a complete history"），否则不可逆。
- 报告里写清「哪些是泄题（已回退）、哪些是合法修复（保留）」，并给出可复核命令。
- 最后跑一次无污染基线，并用它的 reward 与泄漏期对比；这才是唯一能证伪归因的观测。
