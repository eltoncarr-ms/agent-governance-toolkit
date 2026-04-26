# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the blocked_tools feature in GovernancePolicy and PolicyInterceptor."""

from __future__ import annotations

import warnings

import pytest
import yaml

from agent_os.integrations.base import (
    GovernancePolicy,
    PolicyInterceptor,
    ToolCallRequest,
    ToolCallResult,
)
from agent_os.integrations.policy_compose import PolicyHierarchy, compose_policies
from agent_os.integrations.tool_aliases import ToolAliasRegistry


# ---------------------------------------------------------------------------
# Class 1: TestBlockedToolsField
# ---------------------------------------------------------------------------


class TestBlockedToolsField:
    """Tests for the blocked_tools field on GovernancePolicy."""

    def test_default_empty(self) -> None:
        assert GovernancePolicy().blocked_tools == []

    def test_set_explicit(self) -> None:
        p = GovernancePolicy(blocked_tools=["rm", "drop"])
        assert p.blocked_tools == ["rm", "drop"]

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            GovernancePolicy(blocked_tools="not_a_list")  # type: ignore[arg-type]

    def test_invalid_element_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            GovernancePolicy(blocked_tools=[123])  # type: ignore[list-item]

    def test_overlap_warning(self) -> None:
        with pytest.warns(UserWarning, match="appear in both"):
            GovernancePolicy(allowed_tools=["a", "b"], blocked_tools=["b", "c"])


# ---------------------------------------------------------------------------
# Class 2: TestBlockedToolsSerialization
# ---------------------------------------------------------------------------


class TestBlockedToolsSerialization:
    """Round-trip serialization tests for blocked_tools."""

    def test_to_dict_includes_blocked_tools(self) -> None:
        p = GovernancePolicy(blocked_tools=["rm"])
        assert p.to_dict()["blocked_tools"] == ["rm"]

    def test_from_dict_preserves(self) -> None:
        p = GovernancePolicy(blocked_tools=["rm"])
        restored = GovernancePolicy.from_dict(p.to_dict())
        assert restored.blocked_tools == ["rm"]

    def test_to_yaml_includes_blocked_tools(self) -> None:
        p = GovernancePolicy(blocked_tools=["rm"])
        yaml_str = p.to_yaml()
        assert "blocked_tools" in yaml_str
        data = yaml.safe_load(yaml_str)
        assert data["blocked_tools"] == ["rm"]

    def test_from_yaml_preserves(self) -> None:
        p = GovernancePolicy(blocked_tools=["rm"])
        restored = GovernancePolicy.from_yaml(p.to_yaml())
        assert restored.blocked_tools == ["rm"]

    def test_from_dict_missing_field_defaults_empty(self) -> None:
        p = GovernancePolicy.from_dict({"max_tokens": 100})
        assert p.blocked_tools == []

    def test_from_yaml_missing_field_defaults_empty(self) -> None:
        yaml_str = "max_tokens: 100\nmax_tool_calls: 5\n"
        p = GovernancePolicy.from_yaml(yaml_str)
        assert p.blocked_tools == []


# ---------------------------------------------------------------------------
# Class 3: TestBlockedToolsHash
# ---------------------------------------------------------------------------


class TestBlockedToolsHash:
    """Hash behaviour with blocked_tools."""

    def test_different_blocked_tools_different_hash(self) -> None:
        p1 = GovernancePolicy(blocked_tools=["a"])
        p2 = GovernancePolicy(blocked_tools=["b"])
        assert hash(p1) != hash(p2)

    def test_same_blocked_tools_same_hash(self) -> None:
        p1 = GovernancePolicy(blocked_tools=["a", "b"])
        p2 = GovernancePolicy(blocked_tools=["a", "b"])
        assert hash(p1) == hash(p2)


# ---------------------------------------------------------------------------
# Class 4: TestBlockedToolsDiff
# ---------------------------------------------------------------------------


class TestBlockedToolsDiff:
    """diff() behaviour with blocked_tools."""

    def test_diff_detects_blocked_tools_change(self) -> None:
        p1 = GovernancePolicy(blocked_tools=["a"])
        p2 = GovernancePolicy(blocked_tools=["a", "b"])
        d = p1.diff(p2)
        assert "blocked_tools" in d

    def test_diff_no_change(self) -> None:
        p1 = GovernancePolicy(blocked_tools=["a"])
        p2 = GovernancePolicy(blocked_tools=["a"])
        d = p1.diff(p2)
        assert "blocked_tools" not in d


# ---------------------------------------------------------------------------
# Class 5: TestBlockedToolsDetectConflicts
# ---------------------------------------------------------------------------


class TestBlockedToolsDetectConflicts:
    """detect_conflicts() behaviour with blocked_tools."""

    def test_max_zero_with_blocked_tools_warns(self) -> None:
        p = GovernancePolicy(max_tool_calls=0, blocked_tools=["x"])
        conflicts = p.detect_conflicts()
        assert any("no calls permitted anyway" in w for w in conflicts)

    def test_no_conflict_when_empty(self) -> None:
        p = GovernancePolicy()
        conflicts = p.detect_conflicts()
        # No blocked_tools-related conflict should appear
        assert not any("blocked_tools" in w for w in conflicts)


# ---------------------------------------------------------------------------
# Class 6: TestBlockedToolsIsStricterThan
# ---------------------------------------------------------------------------


class TestBlockedToolsIsStricterThan:
    """is_stricter_than() behaviour with blocked_tools."""

    def test_superset_is_stricter(self) -> None:
        assert GovernancePolicy(blocked_tools=["a", "b"]).is_stricter_than(
            GovernancePolicy(blocked_tools=["a"])
        )

    def test_subset_is_not_stricter(self) -> None:
        assert not GovernancePolicy(blocked_tools=["a"]).is_stricter_than(
            GovernancePolicy(blocked_tools=["a", "b"])
        )

    def test_disjoint_is_not_stricter(self) -> None:
        assert not GovernancePolicy(blocked_tools=["b"]).is_stricter_than(
            GovernancePolicy(blocked_tools=["a"])
        )

    def test_same_blocked_not_stricter(self) -> None:
        assert not GovernancePolicy(blocked_tools=["a"]).is_stricter_than(
            GovernancePolicy(blocked_tools=["a"])
        )

    def test_alias_canonicalization(self) -> None:
        # "run_command" and "shell_execute" both canonicalize to "shell_execute"
        registry = ToolAliasRegistry()
        assert registry.canonicalize("run_command") == registry.canonicalize("shell_execute")

        # Blocking ["run_command", "extra"] is a superset of ["shell_execute"]
        assert GovernancePolicy(
            blocked_tools=["run_command", "extra"]
        ).is_stricter_than(GovernancePolicy(blocked_tools=["shell_execute"]))


# ---------------------------------------------------------------------------
# Class 7: TestPolicyInterceptorBlockedTools
# ---------------------------------------------------------------------------


class TestPolicyInterceptorBlockedTools:
    """Tests for PolicyInterceptor.intercept() with blocked_tools."""

    @staticmethod
    def _request(tool_name: str) -> ToolCallRequest:
        return ToolCallRequest(tool_name=tool_name, arguments={})

    def test_blocked_tool_denied(self) -> None:
        policy = GovernancePolicy(blocked_tools=["dangerous_tool"])
        interceptor = PolicyInterceptor(policy)
        result = interceptor.intercept(self._request("dangerous_tool"))
        assert result.allowed is False
        assert "blocked by policy" in (result.reason or "")

    def test_blocked_tool_before_allowed(self) -> None:
        # blocked_tools takes precedence over allowed_tools
        with pytest.warns(UserWarning, match="appear in both"):
            policy = GovernancePolicy(
                allowed_tools=["dangerous_tool"],
                blocked_tools=["dangerous_tool"],
            )
        interceptor = PolicyInterceptor(policy)
        result = interceptor.intercept(self._request("dangerous_tool"))
        assert result.allowed is False

    def test_unblocked_tool_allowed(self) -> None:
        policy = GovernancePolicy(blocked_tools=["bad_tool"])
        interceptor = PolicyInterceptor(policy)
        result = interceptor.intercept(self._request("good_tool"))
        assert result.allowed is True

    def test_empty_blocked_tools_allows_all(self) -> None:
        policy = GovernancePolicy()
        interceptor = PolicyInterceptor(policy)
        result = interceptor.intercept(self._request("any_tool"))
        assert result.allowed is True

    def test_alias_blocked(self) -> None:
        # Blocking "shell_execute" also blocks "run_command" via alias
        policy = GovernancePolicy(blocked_tools=["shell_execute"])
        interceptor = PolicyInterceptor(policy)
        result = interceptor.intercept(self._request("run_command"))
        assert result.allowed is False
        assert "blocked by policy" in (result.reason or "")


# ---------------------------------------------------------------------------
# Class 8: TestBlockedToolsComposition
# ---------------------------------------------------------------------------


class TestBlockedToolsComposition:
    """Tests for policy_compose.py merge behaviour with blocked_tools."""

    def test_compose_unions_blocked_tools(self) -> None:
        p1 = GovernancePolicy(name="p1", blocked_tools=["a"])
        p2 = GovernancePolicy(name="p2", blocked_tools=["b"])
        result = compose_policies(p1, p2)
        assert set(result.blocked_tools) == {"a", "b"}

    def test_compose_deduplicates(self) -> None:
        p1 = GovernancePolicy(name="p1", blocked_tools=["a", "b"])
        p2 = GovernancePolicy(name="p2", blocked_tools=["b", "c"])
        result = compose_policies(p1, p2)
        assert set(result.blocked_tools) == {"a", "b", "c"}

    def test_hierarchy_chain_unions(self) -> None:
        parent = GovernancePolicy(name="parent", blocked_tools=["x"])
        child = GovernancePolicy(name="child", blocked_tools=["y"])
        hierarchy = PolicyHierarchy(parent)
        chained = hierarchy.chain(child)
        assert set(chained.blocked_tools) == {"x", "y"}


# ---------------------------------------------------------------------------
# Class 9: TestBlockedToolsBackwardsCompat
# ---------------------------------------------------------------------------


class TestBlockedToolsBackwardsCompat:
    """Verify no regressions when blocked_tools is absent."""

    def test_old_yaml_without_blocked_tools_loads(self) -> None:
        yaml_str = (
            "max_tokens: 2048\n"
            "max_tool_calls: 5\n"
            "allowed_tools: []\n"
            "require_human_approval: false\n"
        )
        p = GovernancePolicy.from_yaml(yaml_str)
        assert p.blocked_tools == []
        assert p.max_tokens == 2048

    def test_old_dict_without_blocked_tools_loads(self) -> None:
        d = {
            "max_tokens": 2048,
            "max_tool_calls": 5,
            "allowed_tools": [],
            "require_human_approval": False,
        }
        p = GovernancePolicy.from_dict(d)
        assert p.blocked_tools == []

    def test_default_policy_unchanged_behavior(self) -> None:
        p = GovernancePolicy()
        # allows all tools, blocks none
        assert p.allowed_tools == []
        assert p.blocked_tools == []
        interceptor = PolicyInterceptor(p)
        result = interceptor.intercept(
            ToolCallRequest(tool_name="anything", arguments={})
        )
        assert result.allowed is True
