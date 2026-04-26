# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for GovernanceCallbackHandler."""

import sys
import os
import pytest
from unittest.mock import MagicMock, call

# Add the source to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_os.integrations.langchain_callback import (
    GovernanceCallbackHandler,
    _redact_pii,
    _truncate,
)


@pytest.fixture
def mock_audit():
    """Mock GovernanceAuditLogger."""
    audit = MagicMock()
    audit.log_decision = MagicMock()
    audit.flush = MagicMock()
    return audit


@pytest.fixture
def handler(mock_audit):
    return GovernanceCallbackHandler(mock_audit, session_id="test-session")


class TestConstruction:
    def test_construct_with_logger(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1")
        assert handler.audit_logger is mock_audit
        assert handler.session_id == "s1"

    def test_construct_with_none_logger(self):
        handler = GovernanceCallbackHandler(None, session_id="s1")
        assert handler.audit_logger is None


class TestOnToolStart:
    def test_audit_entry_schema(self, handler, mock_audit):
        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="find something",
            run_id="call-1",
        )
        mock_audit.log_decision.assert_called_once()
        kwargs = mock_audit.log_decision.call_args
        assert kwargs[1]["action"] == "search_tool"
        assert kwargs[1]["decision"] == "observe"
        # Check metadata passed as **kwargs
        assert "phase" in {k for k in kwargs[1] if k not in ("agent_id", "action", "decision", "reason", "latency_ms")}

    def test_non_string_payload(self, handler, mock_audit):
        """Dict input should be serialized gracefully."""
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str={"key": "value"},  # Not a string
            run_id="call-2",
        )
        mock_audit.log_decision.assert_called_once()


class TestOnToolEnd:
    def test_audit_entry_schema(self, handler, mock_audit):
        # Start first to record timing
        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="query",
            run_id="call-1",
        )
        mock_audit.reset_mock()
        handler.on_tool_end(output="result data", run_id="call-1")
        mock_audit.log_decision.assert_called_once()
        kwargs = mock_audit.log_decision.call_args[1]
        assert kwargs["action"] == "search_tool"
        assert kwargs["decision"] == "observe"
        assert kwargs["latency_ms"] >= 0

    def test_latency_tracking(self, handler, mock_audit):
        import time
        handler.on_tool_start(
            serialized={"name": "slow_tool"},
            input_str="input",
            run_id="call-lat",
        )
        time.sleep(0.01)  # 10ms
        mock_audit.reset_mock()
        handler.on_tool_end(output="done", run_id="call-lat")
        kwargs = mock_audit.log_decision.call_args[1]
        assert kwargs["latency_ms"] >= 5  # Should be at least ~10ms


class TestOnToolError:
    def test_audit_entry_schema(self, handler, mock_audit):
        handler.on_tool_start(
            serialized={"name": "failing_tool"},
            input_str="input",
            run_id="call-err",
        )
        mock_audit.reset_mock()
        handler.on_tool_error(
            error=ValueError("something broke"),
            run_id="call-err",
        )
        mock_audit.log_decision.assert_called_once()
        kwargs = mock_audit.log_decision.call_args[1]
        assert kwargs["action"] == "failing_tool"
        assert kwargs["decision"] == "error"
        assert "something broke" in kwargs["reason"]


class TestPIIRedaction:
    def test_redact_ssn(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", redact_pii=True)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="SSN is 123-45-6789",
            run_id="r1",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        # The input should contain [REDACTED] not the SSN
        assert "123-45-6789" not in str(kwargs)
        assert "[REDACTED]" in str(kwargs)

    def test_redact_email(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", redact_pii=True)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="email: user@example.com",
            run_id="r2",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        assert "user@example.com" not in str(kwargs)
        assert "[REDACTED]" in str(kwargs)

    def test_redact_credential(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", redact_pii=True)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="password=secret123",
            run_id="r3",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        assert "secret123" not in str(kwargs)
        assert "[REDACTED]" in str(kwargs)

    def test_no_redaction_when_disabled(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", redact_pii=False)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="SSN is 123-45-6789",
            run_id="r4",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        assert "123-45-6789" in str(kwargs)


class TestTruncation:
    def test_redaction_before_truncation(self, mock_audit):
        """Redaction must happen before truncation so PII near the end is caught."""
        # Create input where PII is within first 500 chars
        handler = GovernanceCallbackHandler(
            mock_audit, session_id="s1", max_content_length=500, redact_pii=True
        )
        input_with_pii = "x" * 400 + " SSN: 123-45-6789 " + "y" * 200
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str=input_with_pii,
            run_id="r5",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        assert "123-45-6789" not in str(kwargs)

    def test_truncation_with_ellipsis(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", max_content_length=10)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="a" * 50,
            run_id="r6",
        )
        kwargs = mock_audit.log_decision.call_args[1]
        # Find the input value in metadata
        input_val = str(kwargs)
        assert "..." in input_val

    def test_max_content_length_zero(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1", max_content_length=0)
        handler.on_tool_start(
            serialized={"name": "tool"},
            input_str="some input",
            run_id="r7",
        )
        mock_audit.log_decision.assert_called_once()
        # Entry should still be written but without content
        kwargs = mock_audit.log_decision.call_args[1]
        # No "input" key should be present
        assert "input" not in kwargs or kwargs.get("input", "") == ""


class TestFlush:
    def test_flush_on_success(self, handler, mock_audit):
        handler.on_tool_end(output="result", run_id="r1")
        mock_audit.flush.assert_called()

    def test_flush_on_error(self, handler, mock_audit):
        handler.on_tool_error(error="oops", run_id="r1")
        mock_audit.flush.assert_called()


class TestGracefulDegradation:
    def test_no_writes_with_none_logger(self):
        handler = GovernanceCallbackHandler(None, session_id="s1")
        # Should not raise
        handler.on_tool_start({"name": "tool"}, "input", run_id="r1")
        handler.on_tool_end("output", run_id="r1")
        handler.on_tool_error("error", run_id="r2")

    def test_logger_method_raising(self, mock_audit):
        mock_audit.log_decision.side_effect = RuntimeError("db down")
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1")
        # Should NOT propagate the error
        handler.on_tool_start({"name": "tool"}, "input", run_id="r1")
        handler.on_tool_end("output", run_id="r1")
        handler.on_tool_error("error", run_id="r2")


class TestHandlerReuse:
    def test_no_state_leakage(self, mock_audit):
        handler = GovernanceCallbackHandler(mock_audit, session_id="s1")
        # Tool A lifecycle
        handler.on_tool_start({"name": "tool_a"}, "input_a", run_id="call-a")
        handler.on_tool_end("output_a", run_id="call-a")
        # Tool B lifecycle
        handler.on_tool_start({"name": "tool_b"}, "input_b", run_id="call-b")
        handler.on_tool_end("output_b", run_id="call-b")
        # Verify 4 log_decision calls (start+end for each)
        assert mock_audit.log_decision.call_count == 4
        # After both complete, no stale entries in _call_starts
        assert len(handler._call_starts) == 0


class TestAlongsideOtherHandlers:
    def test_alongside_mock_handler(self, mock_audit):
        """GovernanceCallbackHandler works alongside other handlers."""
        other_handler = MagicMock()
        gov_handler = GovernanceCallbackHandler(mock_audit, session_id="s1")
        
        # Simulate both being called (as LangChain would)
        gov_handler.on_tool_start({"name": "tool"}, "input", run_id="r1")
        other_handler.on_tool_start({"name": "tool"}, "input", run_id="r1")
        
        # Both should have been called
        mock_audit.log_decision.assert_called_once()
        other_handler.on_tool_start.assert_called_once()


class TestHelperFunctions:
    def test_redact_pii_ssn(self):
        assert "[REDACTED]" in _redact_pii("My SSN is 123-45-6789")
        assert "123-45-6789" not in _redact_pii("My SSN is 123-45-6789")

    def test_redact_pii_clean(self):
        assert _redact_pii("no pii here") == "no pii here"

    def test_truncate_short(self):
        assert _truncate("short", 500) == "short"

    def test_truncate_long(self):
        result = _truncate("a" * 600, 500)
        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_truncate_zero(self):
        assert _truncate("anything", 0) == ""
