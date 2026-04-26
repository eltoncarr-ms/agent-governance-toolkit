# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Tests for DX simplification changes (Changes 1, 4, 5).

Run with: python -m pytest tests/test_dx_changes.py -v
"""

import asyncio

import pytest

from agent_os.integrations.base import (
    BaseIntegration,
    ExecutionContext,
    GovernancePolicy,
)


# ---------- helpers ----------

class TestKernel(BaseIntegration):
    """Minimal concrete subclass for testing."""

    def wrap(self, agent):
        return agent

    def unwrap(self, agent):
        return agent


# =====================================================================
# Change 1 – GovernancePolicy.observe_only()
# =====================================================================


def test_observe_only_returns_valid_policy():
    policy = GovernancePolicy.observe_only()
    assert isinstance(policy, GovernancePolicy)


def test_observe_only_field_values():
    policy = GovernancePolicy.observe_only()
    assert policy.confidence_threshold == 0.0
    assert policy.log_all_calls is True
    assert policy.require_human_approval is False
    assert policy.max_tool_calls == 100
    assert policy.max_tokens == 100_000
    assert policy.timeout_seconds == 600


def test_observe_only_overrides():
    policy = GovernancePolicy.observe_only(max_tool_calls=200)
    assert policy.max_tool_calls == 200
    # Other defaults unchanged
    assert policy.confidence_threshold == 0.0


def test_observe_only_name_default():
    policy = GovernancePolicy.observe_only()
    assert policy.name == "observe-only"


def test_observe_only_name_custom():
    policy = GovernancePolicy.observe_only(name="my-obs")
    assert policy.name == "my-obs"


def test_observe_only_behavior_safety_rails():
    """Safety rails (max_tool_calls) are still enforced even in observe mode."""
    policy = GovernancePolicy.observe_only(max_tool_calls=2)
    kernel = TestKernel(policy=policy)
    ctx = kernel.create_context("test-agent")
    ctx.call_count = 2  # At the limit
    allowed, reason = kernel.pre_execute(ctx, "test input")
    assert not allowed
    assert "Max tool calls" in reason


def test_default_constructor_unchanged():
    """Existing GovernancePolicy() default behavior is unchanged."""
    policy = GovernancePolicy()
    assert policy.confidence_threshold == 0.8
    assert policy.max_tool_calls == 10
    assert policy.name == "default"


# =====================================================================
# Change 4 – BaseIntegration.govern()
# =====================================================================


def test_govern_returns_wrapper():
    from agent_os.integrations.base import AsyncGovernedWrapper

    kernel = TestKernel()

    async def my_fn():
        return "result"

    wrapper = kernel.govern(my_fn, agent_id="test")
    assert isinstance(wrapper, AsyncGovernedWrapper)


def test_govern_lifecycle_order():
    kernel = TestKernel()
    call_order = []

    async def my_fn(*args, **kwargs):
        call_order.append("fn")
        return "result"

    wrapper = kernel.govern(my_fn, agent_id="test")
    result = asyncio.get_event_loop().run_until_complete(wrapper())
    assert result == "result"
    assert "fn" in call_order


def test_govern_pre_execute_denial():
    from agent_os.exceptions import PolicyViolationError

    policy = GovernancePolicy(max_tool_calls=1)
    kernel = TestKernel(policy=policy)

    async def my_fn():
        return "result"

    wrapper = kernel.govern(my_fn, agent_id="test")
    # First call should succeed
    asyncio.get_event_loop().run_until_complete(wrapper())
    # Second call should be denied (call_count >= max_tool_calls)
    with pytest.raises(PolicyViolationError, match="Max tool calls"):
        asyncio.get_event_loop().run_until_complete(wrapper())


def test_govern_available_on_subclasses():
    from agent_os.integrations.langchain_adapter import LangChainKernel

    assert hasattr(LangChainKernel, "govern")
    assert callable(getattr(LangChainKernel, "govern"))


# =====================================================================
# Change 5 – YAML roundtrip includes name
# =====================================================================


def test_to_yaml_includes_name():
    policy = GovernancePolicy(name="my-policy")
    yaml_str = policy.to_yaml()
    assert "name: my-policy" in yaml_str


def test_from_yaml_preserves_name():
    yaml_str = "name: my-policy\nmax_tokens: 4096\n"
    policy = GovernancePolicy.from_yaml(yaml_str)
    assert policy.name == "my-policy"


def test_from_dict_preserves_name():
    policy = GovernancePolicy.from_dict({"name": "my-policy"})
    assert policy.name == "my-policy"


def test_save_load_roundtrip_name(tmp_path):
    filepath = str(tmp_path / "policy.yaml")
    original = GovernancePolicy(name="roundtrip-test")
    original.save(filepath)
    loaded = GovernancePolicy.load(filepath)
    assert loaded.name == "roundtrip-test"


def test_from_yaml_no_name_defaults():
    yaml_str = "max_tokens: 4096\n"
    policy = GovernancePolicy.from_yaml(yaml_str)
    assert policy.name == "default"


def test_from_dict_no_name_defaults():
    policy = GovernancePolicy.from_dict({"max_tokens": 4096})
    assert policy.name == "default"
