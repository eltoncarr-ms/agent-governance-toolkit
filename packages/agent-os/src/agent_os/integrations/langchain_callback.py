# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""LangChain callback handler that writes tool-level governance audit entries."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# PII patterns reused from langchain_adapter.py
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),           # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+", re.IGNORECASE),
]


def _redact_pii(text: str) -> str:
    """Replace PII patterns with [REDACTED]."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, appending '...' if truncated."""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


class GovernanceCallbackHandler:
    """LangChain callback handler that writes tool-level audit entries.

    Drop-in alongside any existing callback handler. Writes audit
    entries for tool start, end, and error events.

    This class implements the LangChain callback handler interface
    without requiring langchain-core at import time. It duck-types
    the callback protocol so it works with any LangChain version.

    Args:
        audit_logger: GovernanceAuditLogger instance. When ``None``,
            the handler silently skips all audit writes.
        session_id: Session identifier for audit trail correlation.
        max_content_length: Max chars for input/output in audit entries.
            Default 500. Set to 0 to disable content logging.
        redact_pii: When True (default), redacts PII patterns
            (SSN, email, credentials) from logged content.
    """

    def __init__(
        self,
        audit_logger: Any | None,
        session_id: str = "",
        *,
        max_content_length: int = 500,
        redact_pii: bool = True,
    ) -> None:
        self.audit_logger = audit_logger
        self.session_id = session_id
        self.max_content_length = max_content_length
        self.redact_pii = redact_pii
        self._call_starts: dict[str, tuple[str, float]] = {}  # call_id -> (tool_name, start_time)
        # LangChain callback manager expects these attributes
        self.run_inline = False
        self.raise_error = False
        self.ignore_retry = False
        self.ignore_agent = False
        self.ignore_chain = False
        self.ignore_chat_model = False
        self.ignore_llm = False
        self.ignore_retriever = False
        self.ignore_custom_event = False

    def _process_content(self, content: Any) -> str:
        """Serialize, redact, and truncate content for audit logging."""
        text = str(content) if not isinstance(content, str) else content
        if self.redact_pii:
            text = _redact_pii(text)
        if self.max_content_length > 0:
            text = _truncate(text, self.max_content_length)
        elif self.max_content_length == 0:
            return ""
        return text

    def _write_entry(
        self,
        *,
        action: str,
        decision: str,
        reason: str = "",
        latency_ms: float = 0.0,
        phase: str = "",
        call_id: str = "",
        content_key: str = "",
        content_value: str = "",
    ) -> None:
        """Write an audit entry via the audit logger."""
        if self.audit_logger is None:
            return
        metadata: dict[str, Any] = {
            "phase": phase,
            "call_id": call_id,
            "session_id": self.session_id,
        }
        if content_key and content_value:
            metadata[content_key] = content_value
        try:
            self.audit_logger.log_decision(
                agent_id=self.session_id,
                action=action,
                decision=decision,
                reason=reason,
                latency_ms=latency_ms,
                **metadata,
            )
            self.audit_logger.flush()
        except Exception:
            logger.warning(
                "GovernanceCallbackHandler: audit write failed for %s/%s",
                action, phase,
                exc_info=True,
            )

    # ── LangChain callback interface ──────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "") if isinstance(serialized, dict) else str(serialized)
        call_id = str(run_id) if run_id else str(uuid.uuid4())[:8]
        self._call_starts[call_id] = (tool_name, time.monotonic())

        content = self._process_content(input_str)
        self._write_entry(
            action=tool_name,
            decision="observe",
            phase="tool_start",
            call_id=call_id,
            content_key="input" if content else "",
            content_value=content,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool finishes successfully."""
        call_id = str(run_id) if run_id else ""
        tool_name = ""
        latency_ms = 0.0
        if call_id in self._call_starts:
            tool_name, start_time = self._call_starts.pop(call_id)
            latency_ms = (time.monotonic() - start_time) * 1000

        content = self._process_content(output)
        self._write_entry(
            action=tool_name,
            decision="observe",
            latency_ms=latency_ms,
            phase="tool_end",
            call_id=call_id,
            content_key="result" if content else "",
            content_value=content,
        )

    def on_tool_error(
        self,
        error: BaseException | str,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        call_id = str(run_id) if run_id else ""
        tool_name = ""
        latency_ms = 0.0
        if call_id in self._call_starts:
            tool_name, start_time = self._call_starts.pop(call_id)
            latency_ms = (time.monotonic() - start_time) * 1000

        self._write_entry(
            action=tool_name,
            decision="error",
            reason=str(error),
            latency_ms=latency_ms,
            phase="tool_error",
            call_id=call_id,
        )
