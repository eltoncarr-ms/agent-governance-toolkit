# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Standard governance audit logger with pluggable backends."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A governance audit log entry."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = ""
    agent_id: str = ""
    action: str = ""
    decision: str = ""
    reason: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditBackend(Protocol):
    """Protocol for audit log backends."""

    def write(self, entry: AuditEntry) -> None: ...
    def flush(self) -> None: ...


class JsonlFileBackend:
    """Writes audit entries as JSONL to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, entry: AuditEntry) -> None:
        self._file.write(entry.to_json() + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class InMemoryBackend:
    """Stores audit entries in memory (useful for testing)."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def write(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def flush(self) -> None:
        pass


class LoggingBackend:
    """Writes audit entries via Python logging."""

    def __init__(self, logger_name: str = "agent_os.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    def write(self, entry: AuditEntry) -> None:
        self._logger.info(
            "[%s] agent=%s action=%s decision=%s latency=%.1fms",
            entry.event_type, entry.agent_id, entry.action,
            entry.decision, entry.latency_ms,
        )

    def flush(self) -> None:
        pass


class GovernanceAuditLogger:
    """Standard audit logger with pluggable backends.

    Example::

        audit = GovernanceAuditLogger()
        audit.add_backend(InMemoryBackend())
        audit.log_decision(agent_id="a1", action="search", decision="allow")
    """

    def __init__(self) -> None:
        self._backends: list[Any] = []

    def add_backend(self, backend: Any) -> None:
        self._backends.append(backend)

    def log(self, entry: AuditEntry) -> None:
        for backend in self._backends:
            backend.write(entry)

    def log_decision(
        self,
        agent_id: str,
        action: str,
        decision: str,
        reason: str = "",
        latency_ms: float = 0.0,
        **metadata: Any,
    ) -> None:
        entry = AuditEntry(
            event_type="governance_decision",
            agent_id=agent_id,
            action=action,
            decision=decision,
            reason=reason,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        self.log(entry)

    def flush(self) -> None:
        for backend in self._backends:
            backend.flush()


_BACKEND_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register_builtin_backends() -> None:
    """Populate the registry with built-in and optional backends."""
    _BACKEND_REGISTRY["jsonl"] = lambda **kw: JsonlFileBackend(**kw)
    _BACKEND_REGISTRY["logging"] = lambda **kw: LoggingBackend(**kw)
    _BACKEND_REGISTRY["memory"] = lambda **kw: InMemoryBackend(**kw)

    def _make_otel(**kw: Any) -> Any:
        try:
            from agent_os.otel_audit_backend import OTelLogsBackend
        except ImportError:
            raise ImportError(
                "OTel audit backend requires opentelemetry-sdk. "
                "Install with: pip install 'agent-os-kernel[observability]'"
            ) from None
        return OTelLogsBackend(**kw)

    def _make_azure_monitor(**kw: Any) -> Any:
        try:
            from agent_os.audit_backends.azure_monitor import AzureMonitorBackend
        except ImportError:
            raise ImportError(
                "Azure Monitor audit backend requires azure-monitor-ingestion. "
                "Install with: pip install 'agent-os-kernel[sentinel]'"
            ) from None
        return AzureMonitorBackend(**kw)

    _BACKEND_REGISTRY["otel"] = _make_otel
    _BACKEND_REGISTRY["azure_monitor"] = _make_azure_monitor


def _parse_backend_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Parse backend specs into a backend name and parameter mapping."""
    stripped = spec.strip()
    if ":" not in stripped:
        return stripped, {}

    name, remainder = stripped.split(":", 1)
    params: dict[str, str] = {}
    current_key: str | None = None
    current_value_parts: list[str] = []

    for part in remainder.split(":"):
        if "=" in part:
            if current_key is not None:
                params[current_key] = ":".join(current_value_parts).strip()
            key, value = part.split("=", 1)
            current_key = key.strip()
            current_value_parts = [value]
        elif current_key is not None:
            current_value_parts.append(part)

    if current_key is not None:
        params[current_key] = ":".join(current_value_parts).strip()

    return name.strip(), params


def create_audit_logger(backends: str | None = None) -> GovernanceAuditLogger:
    """Create a :class:`GovernanceAuditLogger` from a backend spec string.

    The *backends* string is a comma-separated list of backend names. Each
    name may include colon-delimited key=value parameters::

        AUDIT_BACKENDS=jsonl:path=/var/log/audit.jsonl,logging,otel
        AUDIT_BACKENDS=azure_monitor:dce_endpoint=https://...:dcr_id=dcr-...:stream_name=Custom-Audit

    When *backends* is ``None``, the ``AUDIT_BACKENDS`` environment variable
    is read. If neither is set, a :class:`LoggingBackend` is used as the
    safe default.

    Args:
        backends: Backend spec string, or ``None`` to read from env.

    Returns:
        Configured :class:`GovernanceAuditLogger`.

    Raises:
        ValueError: If an unknown backend name is encountered.
        ImportError: If a backend's optional dependency is not installed.
    """
    if not _BACKEND_REGISTRY:
        _register_builtin_backends()

    spec = backends if backends is not None else os.environ.get("AUDIT_BACKENDS")
    audit_logger = GovernanceAuditLogger()

    if spec is None:
        audit_logger.add_backend(LoggingBackend())
        return audit_logger

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, params = _parse_backend_spec(item)
        factory = _BACKEND_REGISTRY.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown audit backend '{name}'. "
                f"Available: {', '.join(sorted(_BACKEND_REGISTRY))}"
            )
        audit_logger.add_backend(factory(**params))

    return audit_logger
