"""Tests for PromptBuilder._build_dashboard and _build_memory_keys_summary."""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────

def make_builder():
    """Create a PromptBuilder with a mock SkillManager."""
    from flagscale_agent.react.prompt_builder import PromptBuilder
    skill_mgr = MagicMock()
    skill_mgr.list_skills.return_value = []
    return PromptBuilder(skill_mgr)


# ── _build_dashboard ─────────────────────────────────────────────────────

class TestBuildDashboard:
    def test_turn_always_present(self):
        b = make_builder()
        b._turn_count = 7
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Turn: 7" in result

    def test_no_plan_no_task_step(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Task:" not in result
        assert "Step:" not in result

    def test_plan_title_extracted(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = '<active-plan title="My Task" status="active">'
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Task: My Task" in result

    def test_plan_step_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[🔄] Step 2: in progress\n'
            '[⬜] Step 3: pending\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_plan_step_pending_when_no_doing(self):
        b = make_builder()
        b._turn_count = 1
        plan_ctx = (
            '<active-plan title="T">\n'
            '[✅] Step 1: done\n'
            '[⬜] Step 2: next\n'
            '[⬜] Step 3: later\n'
        )
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard(plan_ctx, session_dir="")
        assert "Step: 2/3" in result

    def test_session_dir_injected(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="/tmp/sess123")
        assert "Session: /tmp/sess123" in result
        assert "conversation.json: /tmp/sess123/conversation.json" in result
        assert "conversation_full.json: /tmp/sess123/conversation_full.json" in result

    def test_no_session_dir_omits_paths(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "conversation.json" not in result
        assert "Session:" not in result

    def test_memory_keys_present(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/a, pitfall/b"):
            result = b._build_dashboard("", session_dir="")
        assert "Memory keys: fact/a, pitfall/b" in result

    def test_empty_memory_keys_omitted(self):
        b = make_builder()
        b._turn_count = 1
        with patch.object(b, "_build_memory_keys_summary", return_value=""):
            result = b._build_dashboard("", session_dir="")
        assert "Memory keys" not in result

    def test_full_dashboard_all_parts(self):
        """All parts present when plan + session + memory all provided."""
        b = make_builder()
        b._turn_count = 5
        plan_ctx = '<active-plan title="Deploy" >\n[🔄] Step 1:\n[⬜] Step 2:\n'
        with patch.object(b, "_build_memory_keys_summary", return_value="fact/x"):
            result = b._build_dashboard(plan_ctx, session_dir="/home/user/.flagscale/sessions/abc")
        assert "Task: Deploy" in result
        assert "Step: 1/2" in result
        assert "Turn: 5" in result
        assert "Session: /home/user/.flagscale/sessions/abc" in result
        assert "Memory keys: fact/x" in result

# ── _build_memory_keys_summary ───────────────────────────────────────────

class TestBuildMemoryKeysSummary:
    def test_returns_keys_only(self, tmp_path):
        """Keys listed, values not included."""
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/cluster/port", "fact", "值: 22")
        mem.put("pitfall/nccl/hang", "pitfall", "现象: hang")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        assert "fact/cluster/port" in result
        assert "pitfall/nccl/hang" in result
        assert "值: 22" not in result
        assert "现象: hang" not in result

    def test_empty_memory_returns_empty_string(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_exception_returns_empty_string(self):
        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.Memory", side_effect=Exception("boom")):
            result = b._build_memory_keys_summary()
        assert result == ""

    def test_multiple_keys_comma_separated(self, tmp_path):
        from flagscale_agent.react.memory import Memory
        mem = Memory(str(tmp_path))
        mem.put("fact/a/b", "fact", "x")
        mem.put("fact/c/d", "fact", "y")
        mem.put("insight/agent/loop", "insight", "z")

        b = make_builder()
        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory", return_value=mem):
            result = b._build_memory_keys_summary()

        keys = [k.strip() for k in result.split(",")]
        assert "fact/a/b" in keys
        assert "fact/c/d" in keys
        assert "insight/agent/loop" in keys


# ── SYSTEM_PROMPT_STATIC content (regression guards) ─────────────────────

class TestSystemPromptContent:
    """Guard against silent removal of failure-loop / cognitive-mode instructions."""

    def _prompt(self):
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        return SYSTEM_PROMPT_STATIC

    def test_stall_is_failure_mode_present(self):
        # Covers regex-chess-style loops: re-analyzing same bug / thinking without acting.
        assert "Stalling is also a failure mode" in self._prompt()

    def test_information_gain_signal_present(self):
        # The stall trigger must be information-gain, not a clock or counter.
        p = self._prompt()
        assert "information gain" in p
        assert "does not depend on a clock or a counter" in p

    def test_protect_working_partial_solution_present(self):
        # Don't let an unsolved detail destroy a working partial solution.
        assert "Do not let an unsolved detail destroy a working partial solution" in self._prompt()

    def test_carry_forward_constraints_present(self):
        # Constraint amnesia: pivoting to clear a blocker must not revert to a proven-bad state.
        p = self._prompt()
        assert "carry forward the constraints you already established" in p
        assert "Re-breaking a known constraint is negative progress" in p

    def test_optimize_first_improvement_not_finish_line_present(self):
        # Optimization tasks: first working improvement is not the finish line;
        # keep pushing across distinct methods when no absolute threshold is given.
        p = self._prompt()
        assert "OPEN-ENDED PROGRESS" in p
        assert "FLOOR, not an achievement" in p
        assert "graded against a reference solution or relative bar you cannot see" in p
        # Baseline-anchor guard: the only informative comparison is between own attempts.
        assert "The only informative comparison is between your own genuinely-different attempts" in p
        # Symmetric other half (path-tracing case): once distinct methods stop yielding
        # gains, tweaking WITHIN one method is burning budget on variance, not progress.
        # Guard both the "two-sided test" framing and the oscillation tell.
        assert '"until improvements stop" is a two-sided test' in p
        assert "burning the budget on variance" in p
        assert "deliver the current best and stop" in p

    def test_noisy_threshold_margin_over_own_measurement_present(self):
        # tune-mjcf rerun: agent's own timing measured the speed ratio just UNDER
        # the pass bar, but the verifier's rigorous measurement (repeats + drop
        # extreme percentiles + mean) landed just OVER it → fail by a hair. The two
        # numbers differ only by the disagreement between measuring instruments.
        # Prompt must name: on a noisy, measurement-dependent, hidden-grader
        # threshold, passing by a hair on your OWN measure is over-fit to your
        # measuring stick; measure the artifact the grader's way and demand margin.
        p = self._prompt()
        assert "NOISY, MEASUREMENT-DEPENDENT metric" in p
        assert "passing by a HAIR according to YOUR OWN measurement" in p
        # robust re-measurement recipe: repeats + discard extreme percentiles + central stat
        assert "discard extreme percentiles" in p
        # margin proportional to noise is the finish condition
        assert "demand MARGIN proportional to the noise" in p
        assert "beat the bar by enough that neither run-to-run variance" in p
        # keep it an observation, not the rhetorical exit
        assert "I re-measured the artifact the way the grader would" in p
        # leak guard: no task-specific nouns from this task
        for w in ("mujoco", "mjcf", "model.xml", "jacobian", "0.60"):
            assert w not in p, f"leaked task-specific term: {w!r}"

    def test_switching_requires_fundamental_difference_present(self):
        assert "Switching methods requires fundamental difference, not variants" in self._prompt()

    def test_classification_gate_present(self):
        assert "CLASSIFICATION GATE" in self._prompt()

    # ── new three-principle structure guards ─────────────────────────────

    def test_three_principles_structure_present(self):
        # The cognitive block is now organized as three principles, each gating
        # a distinct decision moment. Guard against silent collapse back to a flat list.
        p = self._prompt()
        assert "PRINCIPLE 1 — Understand before you implement" in p
        assert "PRINCIPLE 2 — Serve the real goal, do not self-deceive" in p
        assert "PRINCIPLE 3 — When you fail, escape downward or upward, never sideways" in p

    def test_generalize_not_overfit_present(self):
        # Principle 2 core: a scoring check is only a SAMPLE; the goal is a method
        # that generalizes to the real use, not one that overfits the sampled check.
        p = self._prompt()
        assert "only a SAMPLE of a real-world need" in p
        assert "GENERALIZES to the real use" in p

    def test_near_vs_far_end_verification_present(self):
        # Principle 2 core: the subtler self-deception is running an affirmative test
        # that confirms the NEAR end of the chain (command accepted / channel works)
        # and mistaking it for the FAR end (target actually changed state). Verify
        # where the real consumer observes the effect, not the sender's success report.
        p = self._prompt()
        assert "NEAR end of the chain" in p
        assert "FAR end" in p
        assert "the target's own state" in p
        assert "is not \"the input took effect.\"" in p

    def test_single_sample_overfit_magic_constants_present(self):
        # video-processing lesson: developing against ONE visible sample but graded on
        # hidden inputs — passing the sample is the floor, not generalization. Magic
        # constants tuned to the visible sample are the tell; prefer relative/normalized/
        # structure-derived judgments. Task-agnostic (no jump/video/hurdle specifics).
        p = self._prompt()
        assert "calibrating to the one example you can see" in p
        assert "magic constants" in p
        assert "no scale-invariant or structural justification" in p
        # the magic-constant example must stay ABSTRACT — naming specific tuned numbers
        # (e.g. "threshold of 25", "cutoff of 2000", "exceeds 40") re-leaks one task's
        # concrete pipeline. The concept ("an absolute threshold / fixed cutoff") is the
        # insight; the digits are case-by-case noise. Use a generic placeholder N.
        assert "hardcoded \"trigger only when it exceeds N\"" in p
        import re as _re
        for _lit in ("threshold of 25", "cutoff of 2000", "exceeds 40", "rises more than 40"):
            assert _lit not in p, f"leaked concrete tuned magic number: {_lit!r}"
        # openssl+video rerun lesson: the OLD wording ("burden shifts to making the
        # method principled") was a self-defeating escape hatch — video's agent, unable
        # to see the hidden test video, "made the method principled" by arguing each
        # threshold reasonable and stopped there. Tightened so "principled" is only a
        # FIX applied AFTER observing perturbed-input instability, never a substitute
        # for observing because the true label is out of reach.
        assert "cannot confirm the FINAL answer by testing" in p
        assert "never means \"I can't observe my own output's behavior\"" in p
        assert "you took the rhetorical exit" in p
        import re
        for w in ("jump", "hurdle", "takeoff", "jump_analyzer"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)
        # leak guard: the magic-constant example must be an ABSTRACT threshold, not a
        # specific vision/video task's mechanism (pixel threshold, area cutoff). Those
        # concrete nouns are case-by-case leakage of one task's internals.
        for w in ("pixel", "area cutoff", "ffmpeg", "sqlite", "sha256"):
            assert w not in p.lower(), f"leaked task-specific term: {w!r}"
        # domain->tool leak guard: the stress-input passage must NOT hand the agent a
        # domain-to-tool lookup table (e.g. "video -> ffmpeg"). Naming the tool for a
        # domain is case-by-case leakage — the agent must derive the tool from the
        # problem class itself. Only abstract perturbation dimensions may appear.
        assert "ffmpeg" not in p.lower()
        assert "which tool is for you to identify from the problem class, not something to be handed" in p

    def test_observation_vs_argument_litmus_present(self):
        # chess(reward1)/openssl(reward0)/video(reward0) rerun: all three share ONE
        # root — verifying against yourself vs against the consumer's conditions. The
        # decisive tell separating pass from fail is OBSERVATION (chess measured actual
        # pixels) vs ARGUMENT (openssl "runs for me", video "should generalize"). Line85
        # now closes with a single runtime litmus unifying all three near/far forms, plus
        # one general move (reproduce the gap and observe) that needs NO task taxonomy.
        p = self._prompt()
        assert "is my evidence an OBSERVATION" in p
        assert "or an ARGUMENT" in p
        assert "name the GAP between the conditions you developed under" in p
        assert "REPRODUCE that gap and observe the result" in p
        assert "You do not need a taxonomy to know which" in p
        # must stay task-agnostic — no rerun-specific vocabulary leaked in
        import re
        for w in ("chess", "openssl", "cryptography", "hurdle", "jump"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)

    def test_generalization_actionable_moves_present(self):
        # video-processing rerun lesson: agent matched the example's truth but its
        # algorithm HARD-CRASHED (ValueError) on the hidden test input. Line-87 "be
        # principled" was too passive — add two concrete moves: (1) manufacture stress
        # inputs from the one sample to find overfitting early,
        # (2) never hard-crash on hidden input — degrade to a defensible fallback and
        # still emit a well-formed answer. Task-agnostic.
        p = self._prompt()
        assert "MANUFACTURE stress inputs" in p
        assert "WRONG answer versus a HARD CRASH" in p
        assert "degrade to a defensible fallback" in p
        # video rerun #2 lesson: agent READ the perturbation text but only ARGUED
        # ("the algorithm should work as long as...") and never generated a variant
        # input — then hard-crashed on the hidden video. Sharpen: a stress input is a
        # GENERATED artifact run through the actual solution, not a sentence, and its
        # output must be READ. Task-agnostic (domain picks the tool).
        assert "a stress input is NOT a sentence" in p
        assert "NEW INPUT ARTIFACT you GENERATE" in p
        assert "FED THROUGH YOUR ACTUAL SOLUTION" in p
        import re
        for w in ("jump", "hurdle", "takeoff", "jump_analyzer"):
            assert not re.search(r"\b" + w + r"\b", p, re.I)

    def test_verification_environment_parity_present(self):
        # Principle 2: a second form of near/far confusion is environmental —
        # verifying in an environment your own actions contaminated (installed a
        # package, set env var, created a helper) and mistaking "runs for me" for
        # "runs where consumed". Deliverable must carry deps or use only guaranteed
        # target tools; a shipped script must not import a locally pip-installed lib.
        p = self._prompt()
        assert "verification environment must be equivalent to the CONSUMER's environment" in p
        assert "must not import a library you pip-installed only locally" in p
        assert "guaranteed in the environment that will actually run this" in p

    def test_deliverable_addressing_invocation_flavor_present(self):
        # Principle 2: a DISTINCT flavor of the environmental near/far trap — not
        # what you ADDED to the environment but HOW the deliverable gets addressed
        # or invoked. The agent's interactive/login shell resolves the artifact via
        # PATH/rc/alias/cwd; the consumer invokes it bare, as a non-login
        # non-interactive subprocess running the plain command by name. "Runs when I
        # TYPE it" != "runs when a PROGRAM calls it". A found-by-name artifact must
        # live in the target's standard install location, verified by invoking it the
        # bare way the consumer will. Must stay generic (no sqlite/.bashrc/symlink).
        p = self._prompt()
        assert "HOW your deliverable gets addressed or invoked" in p
        assert 'is not "it runs when a PROGRAM calls it."' in p
        assert "standard install location the target guarantees" in p
        assert "primed to find it" in p
        # no case-by-case task specifics leaked into the static prompt
        for w in ("sqlite", ".bashrc", "profile.d", "symlink", "/usr/local/bin"):
            assert w not in p

    def test_deliverable_hygiene_present(self):
        # Principle 2 sub-discipline (c): fixed delivery dir always holds the current
        # best verified version; iterate in scratch, overwrite only after verification.
        p = self._prompt()
        assert "DELIVERABLE HYGIENE" in p
        assert "single source of truth" in p
        # tune-mjcf regression: a better-and-valid candidate held ONLY in memory is
        # not banked; when the process is stopped (timeout) the grader reads the
        # delivery path, not process memory, so an unpersisted in-memory winner
        # scores as if never found. Must prescribe continuous write-through of
        # best-so-far to the delivery path the moment it's measured, not "at the end".
        assert "ONLY in memory" in p or "only in memory" in p
        assert "banked" in p
        assert "write it THROUGH" in p or "write-through" in p.lower()
        assert "Best-so-far lives on disk" in p or "best-so-far lives on disk" in p.lower()
        # names being stopped on timeout/kill/budget as when the in-memory best evaporates
        assert "timeout" in p and ("evaporate" in p or "process" in p)
        # tune-mjcf 4th rerun: when the delivery path IS what gets measured, the agent
        # is forced to overwrite-then-measure; writing a candidate over the verified
        # best then finding it worse and NOT restoring leaves a regression at the
        # delivery path (which ships on timeout). Must prescribe backup + rollback so
        # the delivery path always holds the current best verified version.
        assert "overwrite-then-measure" in p
        assert "BACKUP + ROLLBACK" in p or "backup + rollback" in p.lower()
        assert "restore the backup" in p or "restore" in p.lower()
        assert "strictly better" in p

    def test_deliverable_exact_contents_and_shown_command_byproduct_present(self):
        # single-file-delivery regression: an exact-contents contract (deliver
        # EXACTLY one named file at a fixed path) was failed silently because the
        # agent self-verified by running the example command the TASK SHOWED
        # verbatim, and that command wrote its build byproduct INTO the delivery
        # directory — leaving a stray sibling beside the correct deliverable. The
        # hygiene section must teach both halves: (1) exact-contents means the
        # location holds the named set and NOTHING MORE, a stray sibling fails as
        # hard as a missing deliverable and does so silently; (2) a command the task
        # SHOWS is the consumer's action on your artifact, not a spec for where your
        # byproducts go — running it verbatim can deposit strays; redirect byproducts
        # to scratch or clean them before finishing.
        p = self._prompt()
        low = p.lower()
        assert "EXACT-CONTENTS" in p or "exact-contents" in low
        assert "nothing more" in low
        assert "silently" in low
        assert "shows you" in low
        assert "byproduct" in low
        assert "verbatim" in low
        # stays generic: no nouns from the originating task
        for w in ("polyglot", "cmain", "main.py.c", "fibonacci", ".py.c"):
            assert w not in low, f"leaked task-specific term: {w!r}"

    def test_budget_order_present(self):
        # Principle 2 sub-discipline (d): land a crude complete scorable deliverable
        # at the required path FIRST, refine second — a rough answer that exists beats
        # a perfect one that never got written when the budget runs out mid-task.
        p = self._prompt()
        assert "BUDGET ORDER" in p
        assert "crude" in p
        # the guard against sinking the whole budget with no artifact at the path
        assert "ALL" in p and "that gets scored" in p
        # must not contradict P1 / minimal verification unit
        assert "does NOT contradict" in p or "does not contradict" in p.lower()

    def test_constraint_loyalty_present(self):
        # Principle 2 sub-discipline (a): a stated constraint is task identity;
        # if unmet, report BLOCKED rather than substitute a nearby thing.
        p = self._prompt()
        assert "CONSTRAINT LOYALTY" in p
        assert "part of the task's identity" in p
        # povray lesson: disclosing a substitute is not permission to deliver it,
        # and manufacturing the appearance of satisfaction is still a violation.
        assert "honesty about a deviation is NOT permission" in p
        assert "manufacture the APPEARANCE of satisfaction" in p
        assert "empty honest BLOCKED beats a populated fake" in p

    def test_qualifier_is_machine_checked_present(self):
        # A stated qualifier is verified by an unfakeable external grader — declaring
        # the task done does not move the scored verdict. Examples must name the
        # concrete mechanical checks: version string, byte-for-byte file, time boundary.
        p = self._prompt()
        assert "machine-checked" in p
        assert "byte-for-byte" in p
        # confident self-report does not change the verdict
        assert "not move the scored verdict" in p or "does NOT move the scored verdict" in p
        # failure is unconditional when the qualifier is literally unmet
        assert "the task FAILS" in p

    def test_closed_exemption_list_and_unverified_attribution_present(self):
        # build-cython-ext lesson: a precise list of exceptions is a CLOSED constraint —
        # you may not widen the carve-out to swallow a failure you could not fix. And
        # "pre-existing / external / unrelated" is a claim needing evidence, not a
        # default exit when stuck. Task-agnostic (no pyknotid/planarity specifics).
        p = self._prompt()
        assert "closed constraint" in p
        assert "does not earn you the right to add it to the list" in p
        assert "is a CLAIM that needs evidence" in p
        assert "launder a stuck point into an out-of-scope ruling" in p
        for w in ("pyknotid", "planarity", "reconstructed_space_curve"):
            assert w not in p

    def test_minimal_verification_unit_present(self):
        # Principle 1: prove the one load-bearing assumption in a seconds-scale experiment.
        assert "MINIMAL VERIFICATION UNIT" in self._prompt()

    def test_preserve_irreplaceable_present(self):
        # Principle 1: snapshot/copy an irreplaceable given resource before any action
        # (even innocent-looking exploration) whose side effect could be irreversible.
        p = self._prompt()
        assert "PRESERVE THE IRREPLACEABLE BEFORE YOU TOUCH IT" in p
        assert "copy it first" in p

    def test_network_persistence_info_gain_present(self):
        # Environment Resilience: a single failed web_fetch/search is one failed
        # attempt on one path, NOT a verdict that the info is unavailable. Info gain
        # from external research is high-leverage; do not downgrade to a sample-tuned
        # guess at the first network error. "Could not fetch" needs several distinct
        # failed attempts as evidence, not a single error.
        p = self._prompt()
        assert "information gain" in p
        assert "not a verdict" in p.lower() or "is NOT a verdict" in p
        # must frame giving up early as abandoning the highest-leverage move
        assert "highest-leverage" in p or "high-leverage" in p.lower()
        # "could not fetch" is a claim needing multiple failed attempts as evidence
        assert "claim that needs" in p

    def test_logical_undo_not_byte_restore_present(self):
        # Principle 1 / PRESERVE: reverting a change at the logical level does NOT
        # satisfy a byte/hash immutability check (add-then-remove an internal structure
        # leaves the file's bytes changed). Only never-touching / restore-from-backup is safe.
        p = self._prompt()
        assert "LOGICAL UNDO IS NOT BYTE RESTORE" in p
        assert "hash / checksum / exact-bytes" in p
        # Task-agnostic: the add-then-remove round-trip does not reproduce original bytes,
        # stated abstractly (add an internal structure then remove it), NOT via a specific
        # store/engine. Leak guard below asserts no concrete DB engine name is used here.
        assert "ADD an internal structure and then REMOVE it" in p
        # The only safe strategy: never touch the original / restore from backup.
        assert "never touch the original" in p
        # domain leak guard: the PRESERVE passage must teach copy-first as a general
        # reflex, not via one engine's internals. No specific store/format name may
        # appear (sqlite, wal, sqlite_master, sha256 are all case-by-case leakage).
        for w in ("sqlite", "wal", "sqlite_master", "sha256"):
            assert w not in p.lower()

    def test_known_problem_class_first_move_present(self):
        # Principle 1: name the problem class and use its standard technique before enumerating.
        p = self._prompt()
        assert "KNOWN PROBLEM CLASS" in p
        assert "escape is DOWNWARD" in p or "DOWNWARD" in p

    def test_principle1_task_routing_present(self):
        # Principle 1 must route by task class so it does not force web_fetch on
        # infra/ops work (default_to_action + load_knowledge) nor treat a long
        # training run as a failure signal.
        p = self._prompt()
        assert "ROUTE the task" in p
        assert "INFRASTRUCTURE / OPS" in p
        assert "UNFAMILIAR ALGORITHM" in p
        # Ops minimal verification is a sanity run, and long runtime is not a failure signal.
        assert "SANITY RUN" in p
        assert "long runtime is NOT a failure signal" in p
        # The mandatory web_fetch checkpoint is scoped to novel-problem tasks, not ANY task.
        assert "for any UNFAMILIAR ALGORITHM / NOVEL PROBLEM task" in p


# ── refresh() session_dir passthrough ───────────────────────────────────

class TestRefreshSessionDir:
    def test_session_dir_appears_in_system_prompt(self, tmp_path):
        """refresh() passes session_dir all the way into the system prompt."""
        from flagscale_agent.react.prompt_builder import PromptBuilder

        skill_mgr = MagicMock()
        skill_mgr.list_skills.return_value = []
        builder = PromptBuilder(skill_mgr)

        history = MagicMock()
        captured = {}
        history.set_system_prompt.side_effect = lambda p: captured.__setitem__("prompt", p)

        with patch("flagscale_agent.react.prompt_builder.get_memory_dir", return_value=str(tmp_path)), \
             patch("flagscale_agent.react.prompt_builder.Memory") as mock_mem_cls:
            mock_mem_cls.return_value.list_entries.return_value = []
            builder.refresh(
                history=history,
                active_skill_content={},
                shared_storage_paths=[],
                session_dir="/fake/session/xyz",
            )

        prompt = captured.get("prompt", "")
        assert "/fake/session/xyz" in prompt
        assert "conversation_full.json" in prompt
        assert "conversation.json" in prompt
