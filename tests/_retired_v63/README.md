"""Retired v63 prompt-wording-bound tests.

These files were moved out of the collection on 2026-10-31 as part of the v64
prompt rebuild. They asserted the OLD prompt's verbatim sentences; binding the
new prompt back to those strings is exactly the ratchet that made the prompt
grow monotonically (see docs/prompt_invariants.md). Their load-bearing failure
MODES now live in the v64 prompt body and are covered by the semantic contract
in tests/test_prompt_v64_contract.py.

This directory is NOT collected by pytest (norecursedirs in setup.cfg /
conftest collect_ignore) and is kept only as an auditable archive.
"""
