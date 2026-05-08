# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for BaseIntegration audit logger auto-wiring."""

from __future__ import annotations

from agent_os.audit_logger import GovernanceAuditLogger, InMemoryBackend
from agent_os.integrations.base import BaseIntegration, GovernanceEventType


class _StubIntegration(BaseIntegration):
    """Minimal concrete subclass for testing."""

    def wrap(self, agent):
        return agent

    def unwrap(self, governed_agent):
        return governed_agent


class TestAuditAutoWire:
    """Verify emit() auto-wires to audit logger when configured."""

    def test_emit_writes_audit_entry(self):
        mem = InMemoryBackend()
        audit = GovernanceAuditLogger()
        audit.add_backend(mem)

        integration = _StubIntegration(audit_logger=audit)
        integration.emit(
            GovernanceEventType.POLICY_CHECK,
            {
                "agent_id": "agent-1",
                "phase": "pre_execute",
            },
        )

        assert len(mem.entries) == 1
        entry = mem.entries[0]
        assert entry.event_type == "policy_check"
        assert entry.agent_id == "agent-1"
        assert entry.action == "pre_execute"
        assert entry.decision == "allow"

    def test_emit_without_audit_logger_no_writes(self):
        """Omitting audit_logger preserves original behavior."""
        integration = _StubIntegration()
        integration.emit(GovernanceEventType.POLICY_CHECK, {"agent_id": "a1"})

    def test_all_event_types_mapped(self):
        """All GovernanceEventType values produce valid decisions."""
        mem = InMemoryBackend()
        audit = GovernanceAuditLogger()
        audit.add_backend(mem)

        integration = _StubIntegration(audit_logger=audit)

        expected = {
            GovernanceEventType.POLICY_CHECK: "allow",
            GovernanceEventType.POLICY_VIOLATION: "deny",
            GovernanceEventType.TOOL_CALL_BLOCKED: "deny",
            GovernanceEventType.CHECKPOINT_CREATED: "allow",
            GovernanceEventType.DRIFT_DETECTED: "warn",
        }

        for evt, _decision in expected.items():
            integration.emit(evt, {"agent_id": "a1"})

        assert len(mem.entries) == 5
        for entry, (evt, decision) in zip(mem.entries, expected.items()):
            assert entry.event_type == evt.value
            assert entry.decision == decision

    def test_event_listeners_still_fire(self):
        """Audit logger does not suppress regular event listeners."""
        mem = InMemoryBackend()
        audit = GovernanceAuditLogger()
        audit.add_backend(mem)

        events_received: list[dict] = []
        integration = _StubIntegration(audit_logger=audit)
        integration.on(GovernanceEventType.POLICY_CHECK, lambda data: events_received.append(data))

        data = {"agent_id": "a1", "phase": "pre_execute"}
        integration.emit(GovernanceEventType.POLICY_CHECK, data)

        assert len(events_received) == 1
        assert len(mem.entries) == 1

    def test_audit_error_does_not_crash_emit(self):
        """Audit logger errors are swallowed, not propagated."""

        class _BrokenBackend:
            def write(self, entry):
                raise RuntimeError("backend failure")

            def flush(self):
                return None

        audit = GovernanceAuditLogger()
        audit.add_backend(_BrokenBackend())

        integration = _StubIntegration(audit_logger=audit)
        integration.emit(GovernanceEventType.POLICY_CHECK, {"agent_id": "a1"})

    def test_metadata_passed_through(self):
        """Event data dict is passed as AuditEntry.metadata."""
        mem = InMemoryBackend()
        audit = GovernanceAuditLogger()
        audit.add_backend(mem)

        integration = _StubIntegration(audit_logger=audit)
        data = {"agent_id": "a1", "phase": "pre_execute", "tool": "web_search", "risk": 0.3}
        integration.emit(GovernanceEventType.POLICY_CHECK, data)

        entry = mem.entries[0]
        assert entry.metadata["tool"] == "web_search"
        assert entry.metadata["risk"] == 0.3

    def test_reason_extracted_from_data(self):
        mem = InMemoryBackend()
        audit = GovernanceAuditLogger()
        audit.add_backend(mem)

        integration = _StubIntegration(audit_logger=audit)
        integration.emit(
            GovernanceEventType.POLICY_VIOLATION,
            {
                "agent_id": "a1",
                "reason": "blocked by policy",
            },
        )

        assert mem.entries[0].reason == "blocked by policy"
        assert mem.entries[0].decision == "deny"
