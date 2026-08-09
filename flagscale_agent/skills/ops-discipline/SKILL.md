---
description: General operational discipline for FlagScale infrastructure work. Covers
  reading strategy, shell safety, environment awareness, and root cause diagnosis
  patterns. For training-specific operations, use train-run skill.
name: ops-discipline
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Operational Discipline

General operational rules for infrastructure work. The system prompt covers principles; this skill covers execution details.

---

## Reading strategy — depth over speed

- **Understand before implementing.** For complex tasks, read docs, example configs, and source code BEFORE writing anything.
- **Read complete files, not fragments.** One complete read beats ten partial reads.
- **First read: full file.** Note key line numbers. Subsequent reads: targeted ranges.
- **Record key findings in memory_write** so they survive context compaction.
- **Never re-read a file you read in the last 5 turns** unless it was modified.
- **Breadth matters:** for a training config, read at least: the getting-started doc, an existing example config, and the model's source code.

---

## Shell command rules

- Prefer `grep -rn "pattern" . --include="*.py"` for code search.
- Use `head`/`tail` ONLY for quick previewing. Never truncate error logs you need to diagnose.
- NEVER run the same command twice in a row. If results are unclear, try a DIFFERENT diagnostic.
- NEVER modify third-party source code to work around build errors.
- For large downloads: `wget -c` or `curl -C -`. Execute as SEPARATE commands, not combined with `&&`.
- After any download, verify with `ls -lh <file>`.
- Download speed < 500 KB/s for multi-GB file → check proxy, then STOP and ask user.

---

## Environment awareness

- FIRST thing on any new server: `nvidia-smi`, `cat /etc/os-release`, `which conda`, `echo $CUDA_HOME`. Save to memory.
- Check disk space (`df -h`) before large downloads or builds.
- Check GPU memory (`nvidia-smi`) before launching training.

---

## FlagScale log structure

```
outputs/<exp>/logs/details/host_<N>_<hostname>/<timestamp>/<run_id>/attempt_<N>/<rank>/
  ├── stdout.log   (training metrics, progress)
  └── stderr.log   (errors, warnings, stack traces)
```

**Critical: ALWAYS check stderr.log after launch.** Training can appear "running" while crashing on rank > 0. Use `monitor(output_dir=...)` to auto-scan all stderr files.

---

## Root cause diagnosis

- dtype mismatches (fp32 in bf16 pipelines) are architecture-level. Trace dtype from source rather than adding `.to(dtype)` at error site.
- Cascading TypeError/AttributeError on module init → read the COMPLETE base class API, fix ALL mismatches at once.
- Before calling any base class method, read its IMPLEMENTATION, not just signature.

---

## Fail-fast preflight

Before operations >30 seconds:
- **Model loading**: verify state_dict keys/shapes match BEFORE loading to GPU
- **Checkpoint conversion**: compare key counts/shapes between source and target
- **Training launch**: validate config arithmetic, verify ALL dependencies importable
- **Memory budget**: `params × 2 (bf16) + grads × 2 + optimizer × (8/DP)` — if exceeds GPU memory, don't launch
- **Config arithmetic**: `global_batch_size % (micro_batch_size × DP) == 0`, `num_heads % TP == 0`

---

## Experiment Tracking — HARD GATE

**Every training launch MUST follow this protocol. No exceptions. Not even "quick retries."**

The sequence is: `create` → `add_attempt` → launch → monitor → `update_last_attempt` → (repeat or `finalize`)

| When | Action | Tool Call |
|------|--------|-----------|
| First time working on a model/task | Create experiment | `workspace_experiment(action="create", name=..., purpose=..., hypothesis=...)` |
| BEFORE every `flagscale train` | Record attempt | `workspace_experiment(action="add_attempt", name=..., change=..., config={...}, hardware={...}, output_dir=...)` |
| AFTER every result (success or crash) | Record result | `workspace_experiment(action="update_last_attempt", name=..., result="...")` |
| Done with this experiment line | Close it | `workspace_experiment(action="finalize", name=..., status=..., learnings=[...])` |

**Why this matters:**
- Rapid debug-fix-retry cycles are the HARDEST to reconstruct after the fact
- Without tracking, you repeat failed approaches because you forgot what you tried
- The `change` field forces you to articulate what's different — if you can't, you shouldn't launch
- The `learnings` field in finalize builds institutional knowledge across sessions

**Self-check:** If you're about to call `flagscale train` and haven't called `add_attempt` in this turn, STOP.
