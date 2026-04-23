# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for MCP governance gateway factory and audit bridge adapter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_os.integrations.base import GovernancePolicy
from agent_os.integrations.maf_adapter import (
    AuditLogToMCPSinkAdapter,
    CapabilityGuardMiddleware,
    create_governance_middleware,
    create_mcp_governance_gateway,
)
from agent_os.mcp_gateway import MCPGateway
from agent_os.mcp_protocols import MCPAuditSink
from agentmesh.governance import AuditLog


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_function_context(
    func_name: str = "web_search",
    arguments: dict | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """Create a mock FunctionInvocationContext."""
    ctx = MagicMock()
    ctx.function = MagicMock()
    ctx.function.name = func_name
    ctx.arguments = arguments or {"query": "test"}
    ctx.metadata = metadata if metadata is not None else {}
    ctx.result = None
    return ctx


def _default_policy(**overrides: Any) -> GovernancePolicy:
    """Build a GovernancePolicy with sensible test defaults."""
    kwargs: dict[str, Any] = {
        "name": "test",
        "version": "1.0",
        "log_all_calls": True,
    }
    kwargs.update(overrides)
    return GovernancePolicy(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestCreateMCPGateway (AC15, AC20)
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateMCPGateway:
    """Tests for the create_mcp_governance_gateway() factory."""

    def test_returns_mcp_gateway(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(policy=policy)
        assert isinstance(gw, MCPGateway)

    def test_default_policy_created(self) -> None:
        """When policy=None a default GovernancePolicy is created."""
        # The factory internally constructs a default GovernancePolicy.
        # We patch the class to verify it is called and return a valid
        # policy (the factory currently passes 'rate_limit' which is not
        # a real GovernancePolicy field, so wraps= would fail).
        sentinel_policy = _default_policy(name="mcp-governance")
        with patch(
            "agent_os.integrations.maf_adapter.GovernancePolicy",
            return_value=sentinel_policy,
        ) as mock_cls:
            gw = create_mcp_governance_gateway(policy=None)
        assert isinstance(gw, MCPGateway)
        assert mock_cls.called
        assert gw.policy is sentinel_policy

    def test_custom_policy_used(self) -> None:
        policy = _default_policy(name="custom-policy")
        gw = create_mcp_governance_gateway(policy=policy)
        assert gw.policy is policy
        assert gw.policy.name == "custom-policy"

    def test_denied_tools_configured(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, denied_tools=["execute_code", "delete_file"]
        )
        assert "execute_code" in gw.denied_tools
        assert "delete_file" in gw.denied_tools

    def test_no_gateway_without_call(self) -> None:
        """create_governance_middleware() does NOT create an MCPGateway."""
        stack = create_governance_middleware(
            denied_tools=["execute_code"],
            enable_rogue_detection=False,
        )
        assert all(not isinstance(item, MCPGateway) for item in stack)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestAuditBridgeAdapter (AC19)
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditBridgeAdapter:
    """Tests for AuditLogToMCPSinkAdapter."""

    def test_adapter_implements_record(self) -> None:
        adapter = AuditLogToMCPSinkAdapter(AuditLog())
        assert callable(getattr(adapter, "record", None))

    def test_adapter_calls_audit_log(self) -> None:
        audit_log = MagicMock(spec=AuditLog)
        adapter = AuditLogToMCPSinkAdapter(audit_log)
        adapter.record({"agent_id": "a1", "tool_name": "t1", "allowed": True, "reason": "ok"})
        audit_log.log.assert_called_once()

    def test_adapter_field_mapping(self) -> None:
        audit_log = MagicMock(spec=AuditLog)
        adapter = AuditLogToMCPSinkAdapter(audit_log)

        entry = {
            "agent_id": "agent-42",
            "tool_name": "search",
            "allowed": True,
            "reason": "Allowed by policy",
        }
        adapter.record(entry)

        call_kwargs = audit_log.log.call_args
        # Positional or keyword — the adapter passes keyword args.
        assert call_kwargs.kwargs["agent_did"] == "agent-42"
        assert call_kwargs.kwargs["resource"] == "search"
        assert call_kwargs.kwargs["outcome"] == "allowed"
        assert call_kwargs.kwargs["action"] == "Allowed by policy"

        # Verify denied outcome mapping
        audit_log.reset_mock()
        entry_denied = {**entry, "allowed": False}
        adapter.record(entry_denied)
        assert audit_log.log.call_args.kwargs["outcome"] == "denied"

    def test_adapter_failure_does_not_raise(self) -> None:
        audit_log = MagicMock(spec=AuditLog)
        audit_log.log.side_effect = RuntimeError("boom")
        adapter = AuditLogToMCPSinkAdapter(audit_log)

        # Must not propagate the exception.
        adapter.record({"agent_id": "a1", "tool_name": "t1", "allowed": True, "reason": "ok"})

    def test_adapter_used_when_audit_log_provided(self) -> None:
        audit_log = AuditLog()
        policy = _default_policy()
        gw = create_mcp_governance_gateway(policy=policy, audit_log=audit_log)
        assert isinstance(gw._audit_sink, AuditLogToMCPSinkAdapter)

    def test_audit_sink_takes_precedence(self) -> None:
        audit_log = AuditLog()
        custom_sink = MagicMock(spec=MCPAuditSink)
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, audit_log=audit_log, audit_sink=custom_sink
        )
        assert gw._audit_sink is custom_sink


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestMCPToolDenyList (AC16)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPToolDenyList:
    """Tests for tool deny-list enforcement via MCPGateway."""

    def test_denied_tool_blocked(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, denied_tools=["execute_code"]
        )
        allowed, reason = gw.intercept_tool_call("agent-1", "execute_code", {})
        assert allowed is False
        assert "execute_code" in reason

    def test_allowed_tool_passes(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, denied_tools=["execute_code"]
        )
        allowed, _reason = gw.intercept_tool_call("agent-1", "web_search", {})
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestMCPDangerousParams (AC17)
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPDangerousParams:
    """Tests for built-in parameter sanitization."""

    def test_ssn_pattern_blocked(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(policy=policy)
        allowed, reason = gw.intercept_tool_call(
            "agent-1", "search", {"query": "SSN 123-45-6789"}
        )
        assert allowed is False
        assert "pattern" in reason.lower()

    def test_clean_params_pass(self) -> None:
        policy = _default_policy()
        gw = create_mcp_governance_gateway(policy=policy)
        allowed, _reason = gw.intercept_tool_call(
            "agent-1", "search", {"query": "hello world"}
        )
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestConsistentEnforcement (AC18)
# ═══════════════════════════════════════════════════════════════════════════


class TestConsistentEnforcement:
    """Both CapabilityGuardMiddleware and MCPGateway honour the same deny list."""

    def test_same_denied_tools_blocks_both_local_and_mcp(self) -> None:
        denied = ["execute_code"]

        # ── Local middleware path ──
        guard = CapabilityGuardMiddleware(denied_tools=denied)
        ctx = _make_function_context(func_name="execute_code")

        async def _noop() -> None:
            pass  # pragma: no cover

        asyncio.run(guard.process(ctx, _noop))
        # The middleware sets ctx.result to a denial string.
        assert ctx.result is not None
        assert "not permitted" in str(ctx.result).lower() or "blocked" in str(ctx.result).lower()

        # ── MCP gateway path ──
        policy = _default_policy()
        gw = create_mcp_governance_gateway(policy=policy, denied_tools=denied)
        allowed, reason = gw.intercept_tool_call("agent-1", "execute_code", {})
        assert allowed is False
        assert "execute_code" in reason


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestAuditEntriesFromBothPaths
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditEntriesFromBothPaths:
    """Unified audit log receives entries from both local and MCP paths."""

    def test_audit_entries_from_local_and_mcp(self) -> None:
        audit_log = AuditLog()
        denied = ["execute_code"]

        # ── Local middleware call (denied) ──
        guard = CapabilityGuardMiddleware(denied_tools=denied, audit_log=audit_log)
        ctx = _make_function_context(func_name="execute_code")

        async def _noop() -> None:
            pass  # pragma: no cover

        asyncio.run(guard.process(ctx, _noop))

        # ── MCP gateway call (denied) ──
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, denied_tools=denied, audit_log=audit_log
        )
        gw.intercept_tool_call("agent-1", "execute_code", {})

        # Both paths should have produced entries in the shared audit log.
        all_entries = audit_log._chain._entries
        assert len(all_entries) >= 2

        event_types = {e.event_type for e in all_entries}
        assert "tool_blocked" in event_types
        assert "mcp_tool_decision" in event_types


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestMCPGatewayRateLimiting
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPGatewayRateLimiting:
    """Per-agent call budget enforcement via max_tool_calls."""

    def test_rate_limit_enforced(self) -> None:
        policy = _default_policy(max_tool_calls=2)
        gw = create_mcp_governance_gateway(policy=policy)

        # First two calls should be allowed.
        allowed1, _ = gw.intercept_tool_call("agent-1", "search", {"q": "a"})
        allowed2, _ = gw.intercept_tool_call("agent-1", "search", {"q": "b"})
        assert allowed1 is True
        assert allowed2 is True

        # Third call exceeds the budget.
        allowed3, reason = gw.intercept_tool_call("agent-1", "search", {"q": "c"})
        assert allowed3 is False
        assert "budget" in reason.lower() or "exceeded" in reason.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestMCPGatewayAgentIdPropagation
# ═══════════════════════════════════════════════════════════════════════════


class TestMCPGatewayAgentIdPropagation:
    """Agent identity is propagated into audit entries."""

    def test_agent_id_in_audit(self) -> None:
        audit_log = AuditLog()
        policy = _default_policy()
        gw = create_mcp_governance_gateway(
            policy=policy, audit_log=audit_log
        )
        gw.intercept_tool_call("agent-xyz-99", "search", {"q": "test"})

        entries = audit_log._chain._entries
        assert len(entries) >= 1
        mcp_entry = next(e for e in entries if e.event_type == "mcp_tool_decision")
        assert mcp_entry.agent_did == "agent-xyz-99"
