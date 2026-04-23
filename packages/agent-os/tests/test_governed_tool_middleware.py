# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for GovernedToolMiddleware and its factory integration."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agentmesh", reason="agentmesh not installed")
pytest.importorskip("agent_sre", reason="agent_sre not installed")

from agent_os.integrations.governed_tool import GovernedToolMiddleware, _redact_for_audit
from agent_os.integrations.maf_adapter import create_governance_middleware
from agent_os.policies import PolicyDecision, PolicyEvaluator
from agent_os.policies.schema import (
    PolicyAction,
    PolicyCondition,
    PolicyDefaults,
    PolicyDocument,
    PolicyOperator,
    PolicyRule,
)
from agentmesh.governance import AuditLog


# ── Helpers ──────────────────────────────────────────────────────────────


class _MockFunction:
    def __init__(self, name: str = "test_tool"):
        self.name = name


class _MockFunctionContext:
    def __init__(self, func_name: str = "test_tool", arguments: dict | None = None):
        self.function = _MockFunction(func_name)
        self.arguments = arguments or {}
        self.result = None
        self.metadata: dict[str, Any] = {}


def _make_deny_policy(
    field: str,
    operator: PolicyOperator,
    value: Any,
    *,
    rule_name: str = "test-deny",
    defaults_action: PolicyAction = PolicyAction.ALLOW,
) -> PolicyDocument:
    """Build a one-rule deny policy document."""
    return PolicyDocument(
        name="test-policy",
        rules=[
            PolicyRule(
                name=rule_name,
                condition=PolicyCondition(field=field, operator=operator, value=value),
                action=PolicyAction.DENY,
                message=f"Denied by {rule_name}",
            ),
        ],
        defaults=PolicyDefaults(action=defaults_action),
    )


def _evaluator_with(*docs: PolicyDocument) -> PolicyEvaluator:
    """Return a PolicyEvaluator pre-loaded with the given documents."""
    return PolicyEvaluator(policies=list(docs))


# ── TestRedactForAudit ───────────────────────────────────────────────────


class TestRedactForAudit:
    def test_short_text_unchanged(self):
        assert _redact_for_audit("hello", max_len=200) == "hello"

    def test_long_text_truncated(self):
        text = "x" * 300
        result = _redact_for_audit(text, max_len=200)
        assert result.startswith("x" * 200)
        assert "sha256:" in result
        assert len(result) < len(text) + 30

    def test_hash_is_deterministic(self):
        text = "a" * 500
        r1 = _redact_for_audit(text, max_len=10)
        r2 = _redact_for_audit(text, max_len=10)
        assert r1 == r2
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        assert digest in r1

    def test_empty_string(self):
        assert _redact_for_audit("") == ""


# ── TestToolNameGating ───────────────────────────────────────────────────


class TestToolNameGating:
    @pytest.fixture()
    def deny_delete_evaluator(self) -> PolicyEvaluator:
        doc = _make_deny_policy("tool_name", PolicyOperator.EQ, "delete_file")
        return _evaluator_with(doc)

    @pytest.mark.asyncio
    async def test_deny_by_tool_name(self, deny_delete_evaluator):
        mw = GovernedToolMiddleware(evaluator=deny_delete_evaluator)
        ctx = _MockFunctionContext(func_name="delete_file")
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_not_awaited()
        assert ctx.result is not None
        assert "Policy violation" in ctx.result

    @pytest.mark.asyncio
    async def test_allow_other_tools(self, deny_delete_evaluator):
        mw = GovernedToolMiddleware(evaluator=deny_delete_evaluator)
        ctx = _MockFunctionContext(func_name="web_search")
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_awaited_once()


# ── TestArgumentAwarePolicies ────────────────────────────────────────────


class TestArgumentAwarePolicies:
    @pytest.fixture()
    def deny_ssn_evaluator(self) -> PolicyEvaluator:
        doc = _make_deny_policy(
            "tool_args", PolicyOperator.MATCHES, r"\b\d{3}-\d{2}-\d{4}\b"
        )
        return _evaluator_with(doc)

    @pytest.mark.asyncio
    async def test_deny_ssn_in_args(self, deny_ssn_evaluator):
        mw = GovernedToolMiddleware(evaluator=deny_ssn_evaluator)
        ctx = _MockFunctionContext(
            func_name="send_email",
            arguments={"body": "SSN is 123-45-6789"},
        )
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_not_awaited()
        assert ctx.metadata.get("governance_blocked") is True

    @pytest.mark.asyncio
    async def test_allow_clean_args(self, deny_ssn_evaluator):
        mw = GovernedToolMiddleware(evaluator=deny_ssn_evaluator)
        ctx = _MockFunctionContext(
            func_name="send_email",
            arguments={"body": "Hello, world!"},
        )
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deny_sensitive_path(self):
        doc = _make_deny_policy(
            "tool_args", PolicyOperator.MATCHES, r"/etc/passwd"
        )
        mw = GovernedToolMiddleware(evaluator=_evaluator_with(doc))
        ctx = _MockFunctionContext(
            func_name="read_file",
            arguments={"path": "/etc/passwd"},
        )
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_not_awaited()
        assert ctx.metadata.get("governance_blocked") is True


# ── TestDenialBehavior (AC14a) ───────────────────────────────────────────


class TestDenialBehavior:
    @pytest.fixture()
    def deny_mw(self) -> GovernedToolMiddleware:
        doc = _make_deny_policy("tool_name", PolicyOperator.EQ, "blocked_tool")
        return GovernedToolMiddleware(evaluator=_evaluator_with(doc))

    @pytest.mark.asyncio
    async def test_deny_sets_result(self, deny_mw):
        ctx = _MockFunctionContext(func_name="blocked_tool")
        call_next = AsyncMock()

        await deny_mw.process(ctx, call_next)

        assert ctx.result is not None
        assert ctx.result.startswith("⛔ Policy violation:")

    @pytest.mark.asyncio
    async def test_deny_sets_metadata_flag(self, deny_mw):
        ctx = _MockFunctionContext(func_name="blocked_tool")
        call_next = AsyncMock()

        await deny_mw.process(ctx, call_next)

        assert ctx.metadata["governance_blocked"] is True

    @pytest.mark.asyncio
    async def test_deny_does_not_call_next(self, deny_mw):
        ctx = _MockFunctionContext(func_name="blocked_tool")
        call_next = AsyncMock()

        await deny_mw.process(ctx, call_next)

        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deny_does_not_raise(self, deny_mw):
        ctx = _MockFunctionContext(func_name="blocked_tool")
        call_next = AsyncMock()

        # Should return normally — no exception.
        await deny_mw.process(ctx, call_next)

    @pytest.mark.asyncio
    async def test_allow_calls_next(self, deny_mw):
        ctx = _MockFunctionContext(func_name="safe_tool")
        call_next = AsyncMock()

        await deny_mw.process(ctx, call_next)

        call_next.assert_awaited_once()


# ── TestFailClosed (AC5) ─────────────────────────────────────────────────


class TestFailClosed:
    @pytest.fixture()
    def error_mw(self) -> GovernedToolMiddleware:
        evaluator = PolicyEvaluator()
        evaluator.evaluate = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
        return GovernedToolMiddleware(evaluator=evaluator)

    @pytest.mark.asyncio
    async def test_evaluator_exception_denies(self, error_mw):
        ctx = _MockFunctionContext(func_name="any_tool")
        call_next = AsyncMock()

        await error_mw.process(ctx, call_next)

        call_next.assert_not_awaited()
        assert "Policy violation" in ctx.result

    @pytest.mark.asyncio
    async def test_evaluator_exception_sets_metadata(self, error_mw):
        ctx = _MockFunctionContext(func_name="any_tool")
        call_next = AsyncMock()

        await error_mw.process(ctx, call_next)

        assert ctx.metadata["governance_blocked"] is True

    @pytest.mark.asyncio
    async def test_evaluator_exception_does_not_raise(self, error_mw):
        ctx = _MockFunctionContext(func_name="any_tool")
        call_next = AsyncMock()

        # Must not propagate the exception.
        await error_mw.process(ctx, call_next)


# ── TestDefaultAllow (AC6) ───────────────────────────────────────────────


class TestDefaultAllow:
    @pytest.mark.asyncio
    async def test_empty_rules_allow(self):
        doc = PolicyDocument(
            name="empty",
            rules=[],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        mw = GovernedToolMiddleware(evaluator=_evaluator_with(doc))
        ctx = _MockFunctionContext(func_name="anything")
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_awaited_once()


# ── TestContextExtraction (AC13, AC14b) ──────────────────────────────────


class TestContextExtraction:
    @pytest.fixture()
    def captured_context(self) -> dict[str, Any]:
        return {}

    @pytest.fixture()
    def capture_mw(self, captured_context) -> GovernedToolMiddleware:
        evaluator = PolicyEvaluator()

        original_evaluate = evaluator.evaluate

        def _capturing_evaluate(ctx: dict[str, Any]) -> PolicyDecision:
            captured_context.update(ctx)
            return original_evaluate(ctx)

        evaluator.evaluate = _capturing_evaluate  # type: ignore[assignment]
        return GovernedToolMiddleware(evaluator=evaluator, agent_id="agent-42")

    @pytest.mark.asyncio
    async def test_context_has_tool_name(self, capture_mw, captured_context):
        ctx = _MockFunctionContext(func_name="my_tool")
        await capture_mw.process(ctx, AsyncMock())

        assert captured_context["tool_name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_context_has_tool_args(self, capture_mw, captured_context):
        ctx = _MockFunctionContext(func_name="t", arguments={"a": "1", "b": "2"})
        await capture_mw.process(ctx, AsyncMock())

        assert "tool_args" in captured_context
        assert isinstance(captured_context["tool_args"], str)

    @pytest.mark.asyncio
    async def test_context_has_structured_args(self, capture_mw, captured_context):
        args = {"key": "value"}
        ctx = _MockFunctionContext(func_name="t", arguments=args)
        await capture_mw.process(ctx, AsyncMock())

        assert captured_context["tool_args_structured"] == args

    @pytest.mark.asyncio
    async def test_context_has_agent_id(self, capture_mw, captured_context):
        ctx = _MockFunctionContext(func_name="t")
        await capture_mw.process(ctx, AsyncMock())

        assert captured_context["agent_id"] == "agent-42"

    @pytest.mark.asyncio
    async def test_context_has_no_message_key(self, capture_mw, captured_context):
        ctx = _MockFunctionContext(func_name="t")
        await capture_mw.process(ctx, AsyncMock())

        assert "message" not in captured_context


# ── TestAuditLogging (AC7) ───────────────────────────────────────────────


class TestAuditLogging:
    @pytest.mark.asyncio
    async def test_deny_logs_to_audit(self):
        audit = AuditLog()
        doc = _make_deny_policy("tool_name", PolicyOperator.EQ, "bad_tool")
        mw = GovernedToolMiddleware(
            evaluator=_evaluator_with(doc), audit_log=audit
        )
        ctx = _MockFunctionContext(func_name="bad_tool", arguments={"x": "y" * 500})
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        entries = audit.get_entries_by_type("tool_policy_eval")
        assert len(entries) >= 1
        deny_entry = entries[0]
        assert deny_entry.outcome == "denied"
        # Args should be redacted (truncated) in the log.
        assert "sha256:" in deny_entry.data.get("args_preview", "") or len(
            deny_entry.data.get("args_preview", "")
        ) <= 200

    @pytest.mark.asyncio
    async def test_allow_logs_to_audit(self):
        audit = AuditLog()
        doc = PolicyDocument(
            name="allow-all",
            rules=[],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        mw = GovernedToolMiddleware(
            evaluator=_evaluator_with(doc), audit_log=audit
        )
        ctx = _MockFunctionContext(func_name="safe_tool")
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        entries = audit.get_entries_by_type("tool_policy_eval")
        assert any(e.outcome == "allowed" for e in entries)

    @pytest.mark.asyncio
    async def test_completion_logs_to_audit(self):
        audit = AuditLog()
        doc = PolicyDocument(
            name="allow-all",
            rules=[],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        mw = GovernedToolMiddleware(
            evaluator=_evaluator_with(doc), audit_log=audit
        )
        ctx = _MockFunctionContext(func_name="safe_tool")

        async def _fake_next():
            ctx.result = "tool output here"

        await mw.process(ctx, _fake_next)

        entries = audit.get_entries_by_type("tool_policy_eval")
        completion_entries = [
            e for e in entries if e.data.get("result_preview") is not None
        ]
        assert len(completion_entries) >= 1

    @pytest.mark.asyncio
    async def test_no_audit_when_none(self):
        doc = _make_deny_policy("tool_name", PolicyOperator.EQ, "bad_tool")
        mw = GovernedToolMiddleware(
            evaluator=_evaluator_with(doc), audit_log=None
        )
        ctx = _MockFunctionContext(func_name="bad_tool")
        call_next = AsyncMock()

        # Should not raise even without an audit log.
        await mw.process(ctx, call_next)

        assert ctx.result is not None


# ── TestBackwardCompatibility (AC9, AC10, AC14d) ─────────────────────────


class TestBackwardCompatibility:
    @pytest.fixture()
    def policy_dir(self):
        tmpdir = tempfile.mkdtemp()
        doc = PolicyDocument(
            name="compat-test",
            rules=[],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        doc.to_yaml(os.path.join(tmpdir, "policy.yaml"))
        return tmpdir

    def test_factory_without_flag_excludes_governed_tool(self, policy_dir):
        stack = create_governance_middleware(policy_directory=policy_dir)
        type_names = [type(m).__name__ for m in stack]
        assert "GovernedToolMiddleware" not in type_names

    def test_factory_with_false_excludes_governed_tool(self, policy_dir):
        stack = create_governance_middleware(
            policy_directory=policy_dir, enable_governed_tool_guard=False
        )
        type_names = [type(m).__name__ for m in stack]
        assert "GovernedToolMiddleware" not in type_names

    def test_factory_types_identical(self, policy_dir):
        stack_omit = create_governance_middleware(policy_directory=policy_dir)
        stack_false = create_governance_middleware(
            policy_directory=policy_dir, enable_governed_tool_guard=False
        )
        types_omit = [type(m).__name__ for m in stack_omit]
        types_false = [type(m).__name__ for m in stack_false]
        assert types_omit == types_false


# ── TestFactoryIntegration (AC8, AC14c) ──────────────────────────────────


class TestFactoryIntegration:
    @pytest.fixture()
    def policy_dir(self):
        tmpdir = tempfile.mkdtemp()
        doc = PolicyDocument(
            name="factory-test",
            rules=[
                PolicyRule(
                    name="deny-delete",
                    condition=PolicyCondition(
                        field="tool_name",
                        operator=PolicyOperator.EQ,
                        value="delete_file",
                    ),
                    action=PolicyAction.DENY,
                ),
            ],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        doc.to_yaml(os.path.join(tmpdir, "policy.yaml"))
        return tmpdir

    def test_factory_with_flag_includes_governed_tool(self, policy_dir):
        stack = create_governance_middleware(
            policy_directory=policy_dir, enable_governed_tool_guard=True
        )
        type_names = [type(m).__name__ for m in stack]
        assert "GovernedToolMiddleware" in type_names

    def test_factory_stack_order(self, policy_dir):
        stack = create_governance_middleware(
            policy_directory=policy_dir,
            allowed_tools=["web_search"],
            enable_governed_tool_guard=True,
            enable_rogue_detection=True,
        )
        type_names = [type(m).__name__ for m in stack]
        expected_order = [
            "AuditTrailMiddleware",
            "GovernancePolicyMiddleware",
            "GovernedToolMiddleware",
            "CapabilityGuardMiddleware",
            "RogueDetectionMiddleware",
        ]
        assert type_names == expected_order


# ── TestMixedPolicies (AC14) ─────────────────────────────────────────────


class TestMixedPolicies:
    @pytest.mark.asyncio
    async def test_message_rule_does_not_fire_on_tool_context(self):
        """A rule targeting 'message' should not match tool contexts."""
        doc = PolicyDocument(
            name="mixed-policy",
            rules=[
                PolicyRule(
                    name="block-bad-message",
                    condition=PolicyCondition(
                        field="message",
                        operator=PolicyOperator.MATCHES,
                        value=r"evil",
                    ),
                    action=PolicyAction.DENY,
                    priority=10,
                    message="Message contains evil",
                ),
                PolicyRule(
                    name="block-bad-tool",
                    condition=PolicyCondition(
                        field="tool_name",
                        operator=PolicyOperator.EQ,
                        value="evil_tool",
                    ),
                    action=PolicyAction.DENY,
                    priority=5,
                    message="Tool is evil",
                ),
            ],
            defaults=PolicyDefaults(action=PolicyAction.ALLOW),
        )
        mw = GovernedToolMiddleware(evaluator=_evaluator_with(doc))

        # Call with a non-evil tool — the message rule should not fire
        # because "message" is not in the eval context.
        ctx = _MockFunctionContext(func_name="safe_tool")
        call_next = AsyncMock()

        await mw.process(ctx, call_next)

        call_next.assert_awaited_once()


# ── TestYAMLPolicyLoading (AC11) ─────────────────────────────────────────


class TestYAMLPolicyLoading:
    @pytest.mark.asyncio
    async def test_yaml_policy_loading(self):
        tmpdir = tempfile.mkdtemp()
        yaml_content = """\
version: "1.0"
name: yaml-loaded
rules:
  - name: deny-rm
    condition:
      field: tool_name
      operator: eq
      value: rm_rf
    action: deny
    message: "rm_rf is forbidden"
defaults:
  action: allow
"""
        policy_path = os.path.join(tmpdir, "deny_rm.yaml")
        with open(policy_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        evaluator = PolicyEvaluator()
        evaluator.load_policies(tmpdir)

        mw = GovernedToolMiddleware(evaluator=evaluator)

        # Denied tool.
        ctx_deny = _MockFunctionContext(func_name="rm_rf")
        call_next_deny = AsyncMock()
        await mw.process(ctx_deny, call_next_deny)
        call_next_deny.assert_not_awaited()
        assert ctx_deny.metadata.get("governance_blocked") is True

        # Allowed tool.
        ctx_allow = _MockFunctionContext(func_name="ls")
        call_next_allow = AsyncMock()
        await mw.process(ctx_allow, call_next_allow)
        call_next_allow.assert_awaited_once()
