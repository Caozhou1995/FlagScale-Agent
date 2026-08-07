"""Smoke test: agent can be fully imported and instantiated without errors.

This catches import-time failures (missing modules, circular imports, 
removed guards still referenced) that would crash on reload.
"""
import pytest


def test_agent_import():
    """All imports in agent.py resolve without errors."""
    from flagscale_agent.react.agent import WorkerAgent
    assert WorkerAgent is not None


def test_guard_registry_complete():
    """All guards imported in agent.py can be instantiated."""
    from flagscale_agent.react.guard.safety import ShellSafetyGuard
    from flagscale_agent.react.guard.loop_detect import LoopDetectGuard
    from flagscale_agent.react.guard.progress import ProgressGuard
    from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
    from flagscale_agent.react.guard.plan import PlanGuard
    from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard
    from flagscale_agent.react.guard.constraint import ConstraintGuard
    from flagscale_agent.react.guard.output_dir_reuse import OutputDirReuseGuard
    from flagscale_agent.react.guard.package_search import PackageSearchGuard
    from flagscale_agent.react.guard.debug_discipline import DebugDisciplineGuard
    from flagscale_agent.react.guard.file_tool import FileToolGuard
    from flagscale_agent.react.guard.unit_test import UnitTestGuard
    from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
    from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard
    from flagscale_agent.react.guard.knowledge_first import KnowledgeFirstGuard
    from flagscale_agent.react.guard.arg_type import ArgTypeGuard
    from flagscale_agent.react.guard.error_classifier import ErrorClassifierGuard

    # All must instantiate without error (no missing deps)
    ShellSafetyGuard()
    LoopDetectGuard()
    ProgressGuard()
    ContextPressureGuard()
    PlanGuard()
    TrainingMonitorGuard()
    ConstraintGuard()
    OutputDirReuseGuard()
    PackageSearchGuard()
    DebugDisciplineGuard()
    FileToolGuard()
    UnitTestGuard()
    MemoryDisciplineGuard()
    PostEvictRecoveryGuard()
    KnowledgeFirstGuard()
    ErrorClassifierGuard()


def test_tool_registry_complete():
    """All tools imported in agent.py can be instantiated."""
    from flagscale_agent.react.tools.shell import ShellTool
    from flagscale_agent.react.tools.read_file import ReadFileTool
    from flagscale_agent.react.tools.write_file import WriteFileTool
    from flagscale_agent.react.tools.edit_file import EditFileTool
    from flagscale_agent.react.tools.web_fetch import WebFetchTool
    from flagscale_agent.react.tools.load_skill import LoadSkillTool
    from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool
    from flagscale_agent.react.tools.monitor import FlagScaleTrainMonitorTool
    from flagscale_agent.react.tools.inspect_checkpoint import InspectCheckpointTool
    from flagscale_agent.react.tools.evict import EvictTool
    from flagscale_agent.react.tools.evict_list import EvictListTool
    from flagscale_agent.react.tools.recall import RecallTool
    from flagscale_agent.react.tools.memory_write import MemoryWriteTool
    from flagscale_agent.react.tools.memory_read import MemoryReadTool
    from flagscale_agent.react.tools.memory_list import MemoryListTool
    from flagscale_agent.react.tools.plan_create import PlanCreateTool
    from flagscale_agent.react.tools.plan_status import PlanStatusTool

    # Verify imports resolve (some tools need constructor args, so just check class exists)
    assert ShellTool is not None
    assert ReadFileTool is not None
    assert WriteFileTool is not None
    assert EditFileTool is not None
    assert WebFetchTool is not None
    assert LoadSkillTool is not None
    assert LoadKnowledgeTool is not None
    assert FlagScaleTrainMonitorTool is not None
    assert InspectCheckpointTool is not None
    assert EvictTool is not None
    assert EvictListTool is not None
    assert RecallTool is not None
    assert MemoryWriteTool is not None
    assert MemoryReadTool is not None
    assert MemoryListTool is not None
    assert PlanCreateTool is not None
    assert PlanStatusTool is not None
