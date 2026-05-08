# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Azure Monitor Logs backend for governance audit entries."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from agent_os.audit_logger import AuditEntry

logger = logging.getLogger(__name__)

_HAS_AZURE_MONITOR = False
_LogsIngestionClient: Any = None
_DefaultAzureCredential: Any = None

try:
    from azure.identity import DefaultAzureCredential as _DAC
    from azure.monitor.ingestion import LogsIngestionClient as _LIC

    _HAS_AZURE_MONITOR = True
    _LogsIngestionClient = _LIC
    _DefaultAzureCredential = _DAC
except ImportError:  # pragma: no cover
    pass


class AzureMonitorBackend:
    """Batch governance audit entries to Azure Monitor Logs.

    Args:
        dce_endpoint: Azure Monitor Data Collection Endpoint URL.
        dcr_id: Data Collection Rule immutable ID.
        stream_name: Target stream name configured in the DCR.
        credential: Azure credential instance. Uses ``DefaultAzureCredential``
            when not provided.
        batch_size: Number of records to buffer before auto-flushing.
        flush_interval: Max seconds between flush attempts during writes.
    """

    def __init__(
        self,
        dce_endpoint: str,
        dcr_id: str,
        stream_name: str,
        credential: Any | None = None,
        batch_size: int = 50,
        flush_interval: float = 30.0,
    ) -> None:
        self._enabled = _HAS_AZURE_MONITOR
        self._client: Any = None
        self._dcr_id = dcr_id
        self._stream_name = stream_name
        self._batch_size = max(1, batch_size)
        self._flush_interval = max(0.0, flush_interval)
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.monotonic()

        if not _HAS_AZURE_MONITOR:
            logger.debug(
                "Azure Monitor SDK not installed — AzureMonitorBackend disabled. "
                "Install with: pip install 'agent-os-kernel[sentinel]'"
            )
            return

        try:
            resolved_credential = credential
            if resolved_credential is None:
                credential_cls = _DefaultAzureCredential
                if credential_cls is None:
                    from azure.identity import DefaultAzureCredential as credential_cls
                resolved_credential = credential_cls()
            self._client = _LogsIngestionClient(
                endpoint=dce_endpoint,
                credential=resolved_credential,
            )
        except Exception:
            logger.debug(
                "Failed to initialize Azure Monitor client — AzureMonitorBackend disabled",
                exc_info=True,
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Return whether the backend is ready to upload records."""
        return self._enabled and self._client is not None

    def write(self, entry: AuditEntry) -> None:
        """Buffer an audit entry and flush when thresholds are reached."""
        if not self.enabled:
            return

        payload: list[dict[str, Any]] | None = None
        now = time.monotonic()
        with self._lock:
            self._buffer.append(self._entry_to_record(entry))
            should_flush = len(self._buffer) >= self._batch_size
            if not should_flush and self._flush_interval > 0:
                should_flush = (now - self._last_flush_time) >= self._flush_interval
            if should_flush:
                payload = list(self._buffer)
                self._buffer.clear()
                self._last_flush_time = now

        if payload:
            self._upload(payload)

    def flush(self) -> None:
        """Upload any buffered audit entries to Azure Monitor."""
        if not self.enabled:
            return

        payload: list[dict[str, Any]] = []
        with self._lock:
            if self._buffer:
                payload = list(self._buffer)
                self._buffer.clear()
                self._last_flush_time = time.monotonic()

        if payload:
            self._upload(payload)

    def close(self) -> None:
        """Flush buffered entries and close the Azure client."""
        self.flush()
        if not self.enabled:
            return
        try:
            self._client.close()
        except Exception:
            logger.debug("Failed to close Azure Monitor client", exc_info=True)

    def _entry_to_record(self, entry: AuditEntry) -> dict[str, Any]:
        """Convert an audit entry to a flat Azure Monitor record."""
        return {
            "TimeGenerated": entry.timestamp,
            "EventType": entry.event_type,
            "AgentId": entry.agent_id,
            "Action": entry.action,
            "Decision": entry.decision,
            "Reason": entry.reason,
            "LatencyMs": entry.latency_ms,
            "Metadata": json.dumps(entry.metadata, default=str),
        }

    def _upload(self, payload: list[dict[str, Any]]) -> None:
        """Upload a batch of records, logging failures without raising."""
        try:
            self._client.upload(
                rule_id=self._dcr_id,
                stream_name=self._stream_name,
                logs=payload,
            )
        except Exception:
            logger.exception("Failed to upload governance audit entries to Azure Monitor")
