# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for blocked_tools enforcement in LangChainKernel._check_tool_policy()."""

import pytest

from agent_os.integrations.base import GovernancePolicy
from agent_os.integrations.langchain_adapter import LangChainKernel, PolicyViolationError


class TestLangChainBlockedTools:
    """Verify that the blocked_tools deny-list is enforced correctly."""

    def test_blocked_tool_raises_violation(self) -> None:
        policy = GovernancePolicy(blocked_tools=["dangerous_tool"])
        kernel = LangChainKernel(policy=policy)

        with pytest.raises(PolicyViolationError, match="blocked by policy"):
            kernel._check_tool_policy("dangerous_tool", (), {}, None)

    def test_unblocked_tool_passes(self) -> None:
        policy = GovernancePolicy(blocked_tools=["dangerous_tool"])
        kernel = LangChainKernel(policy=policy)

        # Should not raise
        kernel._check_tool_policy("safe_tool", (), {}, None)

    def test_blocked_before_allowed(self) -> None:
        with pytest.warns(UserWarning, match="blocked_tools takes precedence"):
            policy = GovernancePolicy(
                allowed_tools=["x"], blocked_tools=["x"]
            )

        kernel = LangChainKernel(policy=policy)

        with pytest.raises(PolicyViolationError, match="blocked by policy"):
            kernel._check_tool_policy("x", (), {}, None)

    def test_empty_blocked_tools_no_effect(self) -> None:
        policy = GovernancePolicy()
        kernel = LangChainKernel(policy=policy)

        # Should not raise
        kernel._check_tool_policy("any_tool", (), {}, None)

    def test_alias_blocked(self) -> None:
        # "run_command" canonicalizes to "shell_execute" via DEFAULT_ALIASES
        policy = GovernancePolicy(blocked_tools=["shell_execute"])
        kernel = LangChainKernel(policy=policy)

        with pytest.raises(PolicyViolationError, match="blocked by policy"):
            kernel._check_tool_policy("run_command", (), {}, None)

    def test_blocked_does_not_affect_pattern_check(self) -> None:
        policy = GovernancePolicy(
            blocked_tools=["tool_a"],
            blocked_patterns=["DROP TABLE"],
        )
        kernel = LangChainKernel(policy=policy)

        # tool_b is NOT blocked, but args match a blocked pattern
        with pytest.raises(PolicyViolationError, match="Blocked pattern"):
            kernel._check_tool_policy("tool_b", ("DROP TABLE",), {}, None)
