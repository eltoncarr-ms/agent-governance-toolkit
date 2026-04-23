# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Policy-evaluated FunctionMiddleware for tool governance.

Evaluates every tool invocation against :class:`PolicyEvaluator` rules
(YAML-based governance) and blocks calls that violate policy.  Allowed
invocations are forwarded to the next middleware; denied invocations
receive a human-readable error string as the function result so the LLM
can respond naturally.

This middleware complements :class:`CapabilityGuardMiddleware` (static
allow/deny lists) by supporting rich, declarative policy rules including
regex matching, conditional logic, and external backends.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Awaitable, Callable

from agent_os.policies import PolicyDecision, PolicyEvaluator
from agentmesh.governance import AuditLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional MAF imports — fall back to local stubs when agent_framework
# is not installed so the module remains importable for testing / linting.
# ---------------------------------------------------------------------------
try:
    from agent_framework import FunctionInvocationContext, FunctionMiddleware
except ImportError:  # pragma: no cover
    logger.debug(
        "agent_framework is not installed; GovernedToolMiddleware will use "
        "protocol-only base stubs."
    )

    class FunctionMiddleware:  # type: ignore[no-redef]
        """Stub base class when agent_framework is absent."""

    class FunctionInvocationContext:  # type: ignore[no-redef]
        """Stub for type hints."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_for_audit(text: str, max_len: int = 200) -> str:
    """Truncate *text* for audit logs, appending a content hash when trimmed."""
    if len(text) <= max_len:
        return text
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return text[:max_len] + f"...[sha256:{digest}]"


# ---------------------------------------------------------------------------
# GovernedToolMiddleware
# ---------------------------------------------------------------------------


class GovernedToolMiddleware(FunctionMiddleware):
    """FunctionMiddleware that evaluates tool calls against policy rules.

    Each tool invocation is evaluated against loaded
    :class:`~agent_os.policies.PolicyEvaluator` rules.  If the policy
    denies the call, the function result is set to an error string and
    the middleware returns without calling the next handler.  This keeps
    the tool call intact in the session so the LLM receives the denial
    as tool output and can respond naturally.

    The evaluator is **fail-closed**: if the evaluator raises an
    exception, the call is denied with a generic error message.

    Args:
        evaluator: Pre-configured :class:`PolicyEvaluator` with loaded
            policy documents.
        agent_id: Identifier for the calling agent, used in audit
            entries and evaluation context.
        audit_log: Optional :class:`AuditLog` for recording decisions.
    """

    def __init__(
        self,
        evaluator: PolicyEvaluator,
        agent_id: str = "default-agent",
        audit_log: AuditLog | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.agent_id = agent_id
        self.audit_log = audit_log

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Evaluate governance policy before tool execution."""
        func_name: str = getattr(
            getattr(context, "function", None), "name", "unknown"
        )
        args_dict: dict[str, Any] = getattr(context, "arguments", {}) or {}
        flattened_args: str = " ".join(str(v) for v in args_dict.values())

        # Build context dict for the policy evaluator.
        # Intentionally omit "message" to prevent message-level rules
        # from cross-firing on tool invocations.
        eval_context: dict[str, Any] = {
            "tool_name": func_name,
            "tool_args": flattened_args,
            "tool_args_structured": args_dict,
            "agent_id": self.agent_id,
        }

        # Evaluate — fail-closed on errors.
        try:
            decision: PolicyDecision = self.evaluator.evaluate(eval_context)
        except Exception:
            logger.exception(
                "Policy evaluator error for tool '%s'; denying (fail-closed)",
                func_name,
            )
            context.result = (
                "⛔ Policy violation: internal policy evaluation error"
            )
            metadata: dict[str, Any] = getattr(context, "metadata", {})
            metadata["governance_blocked"] = True
            if self.audit_log:
                self.audit_log.log(
                    event_type="tool_policy_eval",
                    agent_did=self.agent_id,
                    action="error",
                    resource=func_name,
                    data={
                        "tool": func_name,
                        "args_preview": _redact_for_audit(flattened_args),
                        "error": "evaluator_exception",
                    },
                    outcome="denied",
                )
            return

        if not decision.allowed:
            logger.info(
                "Policy DENY for tool '%s': %s (rule=%s)",
                func_name,
                decision.reason,
                decision.matched_rule,
            )

            context.result = f"⛔ Policy violation: {decision.reason}"

            metadata = getattr(context, "metadata", {})
            metadata["governance_blocked"] = True

            if self.audit_log:
                self.audit_log.log(
                    event_type="tool_policy_eval",
                    agent_did=self.agent_id,
                    action=decision.action,
                    resource=func_name,
                    data={
                        "tool": func_name,
                        "args_preview": _redact_for_audit(flattened_args),
                        "matched_rule": decision.matched_rule,
                        "reason": decision.reason,
                    },
                    outcome="denied",
                )

            return

        # Policy allowed — log and continue the pipeline.
        logger.debug(
            "Policy ALLOW for tool '%s' (rule=%s)",
            func_name,
            decision.matched_rule,
        )

        if self.audit_log:
            self.audit_log.log(
                event_type="tool_policy_eval",
                agent_did=self.agent_id,
                action=decision.action,
                resource=func_name,
                data={
                    "tool": func_name,
                    "args_preview": _redact_for_audit(flattened_args),
                    "matched_rule": decision.matched_rule,
                    "reason": decision.reason,
                },
                outcome="allowed",
            )

        await call_next()

        # Log completion with a redacted result preview.
        result_preview = _redact_for_audit(
            str(getattr(context, "result", ""))
        )
        if self.audit_log:
            self.audit_log.log(
                event_type="tool_policy_eval",
                agent_did=self.agent_id,
                action="complete",
                resource=func_name,
                data={
                    "tool": func_name,
                    "result_preview": result_preview,
                },
                outcome="allowed",
            )
