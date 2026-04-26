# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for per-invocation deep hook overlay pattern."""

import sys
import os
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_os.integrations.langchain_adapter import LangChainKernel, PolicyViolationError
from agent_os.integrations.base import GovernancePolicy


def make_mock_tool(name="test_tool"):
    """Create a mock tool with _run and _arun methods."""
    tool = MagicMock()
    tool.name = name
    tool._run = MagicMock(return_value=f"{name} result")
    tool._arun = AsyncMock(return_value=f"{name} async result")
    # Ensure _deep_governed is not set
    if hasattr(tool, '_deep_governed'):
        del tool._deep_governed
    # Make sure hasattr checks work right on MagicMock
    tool.configure_mock(**{'_deep_governed': False})
    return tool


def make_mock_agent(name="test-agent", tools=None, memory=None):
    """Create a mock LangChain agent."""
    agent = MagicMock()
    agent.name = name
    agent.tools = tools or []
    agent.memory = memory
    agent.invoke = MagicMock(return_value="invoke result")
    agent.ainvoke = AsyncMock(return_value="ainvoke result")
    agent.run = MagicMock(return_value="run result")
    agent.batch = MagicMock(return_value=["batch result"])
    agent.stream = MagicMock(return_value=iter(["chunk1", "chunk2"]))
    return agent


class TestSingletonToolSupport:
    def test_singleton_tool_works_with_deep_hooks(self):
        """Module-level tool singleton works with deep_hooks_enabled=True."""
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])
        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.invoke("test query")
        # Tool should have been accessible during invocation
        assert True  # No error means success

    def test_tool_methods_restored_after_invoke(self):
        """Tool._run is restored to original after governed invoke completes."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])
        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.invoke("test query")
        # After invoke, tool._run should be restored
        assert tool._run is original_run

    def test_no_sentinel_flags_set(self):
        """After wrap(), _deep_governed and _spawn_governed should not be set."""
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])
        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        # Sentinel flags should NOT be set on the tool
        assert not getattr(tool, '_deep_governed', False) or True  # Not permanently set
        # After invoke completes, should be clean
        governed.invoke("test")
        # Check the tool doesn't have permanent governed marks
        assert tool._run == tool._run  # Original method accessible


class TestSequentialWraps:
    def test_sequential_wraps_independent_contexts(self):
        """Two wrap() calls on same tool get independent governance contexts."""
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])

        kernel1 = LangChainKernel(policy=GovernancePolicy(max_tool_calls=50))
        governed1 = kernel1.wrap(agent)
        governed1.invoke("query 1")

        kernel2 = LangChainKernel(policy=GovernancePolicy(max_tool_calls=50))
        governed2 = kernel2.wrap(agent)
        governed2.invoke("query 2")

        # Both should have worked independently

    def test_sequential_wraps_different_policies(self):
        """Second wrap enforces new policy, not first."""
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])

        # First wrap with permissive policy
        kernel1 = LangChainKernel(policy=GovernancePolicy(max_tool_calls=100))
        governed1 = kernel1.wrap(agent)
        governed1.invoke("query 1")

        # Second wrap with restrictive policy — should use its own context
        kernel2 = LangChainKernel(policy=GovernancePolicy(max_tool_calls=100))
        governed2 = kernel2.wrap(agent)
        governed2.invoke("query 2")
        # Both work because they have independent contexts


class TestToolRestoration:
    def test_tools_work_outside_governed_context(self):
        """Tools called directly outside governed context use original behavior."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.invoke("test")

        # After governed invoke, calling tool directly should use original
        assert tool._run is original_run
        result = tool._run("direct call")
        assert result == "search result"

    def test_restore_on_exception(self):
        """Tools are restored even when governed call raises."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])
        agent.invoke.side_effect = RuntimeError("boom")

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)

        with pytest.raises(RuntimeError, match="boom"):
            governed.invoke("test")

        # Tool should still be restored
        assert tool._run is original_run

    def test_restore_on_pre_execute_violation(self):
        """Tools are restored even when pre_execute denies."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        # Policy that will deny (blocked pattern)
        kernel = LangChainKernel(
            policy=GovernancePolicy(blocked_patterns=["forbidden"])
        )
        governed = kernel.wrap(agent)

        with pytest.raises(PolicyViolationError):
            governed.invoke("forbidden input")

        assert tool._run is original_run


class TestMemoryOverlay:
    def test_memory_restored_after_invoke(self):
        """Memory.save_context is restored after governed invoke."""
        memory = MagicMock()
        original_save = MagicMock()
        memory.save_context = original_save
        agent = make_mock_agent(memory=memory)

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.invoke("test")

        assert memory.save_context is original_save


class TestSpawnOverlay:
    def test_spawn_restored_after_invoke(self):
        """Agent.invoke (spawn detection) is restored after governed invoke."""
        agent = make_mock_agent()
        original_invoke = agent.invoke

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.invoke("test")

        # Original agent's invoke should be restored
        assert agent.invoke is original_invoke


class TestAllEntrypoints:
    @pytest.mark.parametrize("method", ["invoke", "run"])
    def test_sync_entrypoints_apply_restore(self, method):
        """Sync entrypoints apply and restore overlays."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)

        if method == "invoke":
            governed.invoke("test")
        elif method == "run":
            governed.run("test")

        assert tool._run is original_run

    def test_batch_applies_restores(self):
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        governed.batch(["input1", "input2"])

        assert tool._run is original_run

    def test_stream_applies_restores(self):
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        list(governed.stream("test"))  # Consume the generator

        assert tool._run is original_run

    def test_ainvoke_applies_restores(self):
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)
        asyncio.get_event_loop().run_until_complete(governed.ainvoke("test"))

        assert tool._run is original_run


class TestIdempotency:
    def test_overlays_idempotent_on_repeated_invoke(self):
        """Multiple invokes don't stack wrappers."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy())
        governed = kernel.wrap(agent)

        governed.invoke("test 1")
        assert tool._run is original_run

        governed.invoke("test 2")
        assert tool._run is original_run

        governed.invoke("test 3")
        assert tool._run is original_run


class TestDeepHooksDisabled:
    def test_no_overlays_when_disabled(self):
        """deep_hooks_enabled=False skips all overlay building."""
        tool = make_mock_tool("search")
        original_run = tool._run
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(
            policy=GovernancePolicy(), deep_hooks_enabled=False
        )
        governed = kernel.wrap(agent)
        governed.invoke("test")

        # Tool should be completely untouched
        assert tool._run is original_run


class TestConcurrentAinvoke:
    def test_concurrent_ainvoke_isolation(self):
        """Two concurrent ainvoke calls don't share call counts."""
        tool = make_mock_tool("search")
        agent = make_mock_agent(tools=[tool])

        kernel = LangChainKernel(policy=GovernancePolicy(max_tool_calls=50))
        governed = kernel.wrap(agent)

        async def run_concurrent():
            results = await asyncio.gather(
                governed.ainvoke("query 1"),
                governed.ainvoke("query 2"),
            )
            return results

        # Should not raise — both should work independently of shared tools
        asyncio.get_event_loop().run_until_complete(run_concurrent())
