# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the Azure Monitor audit backend."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pytest

from agent_os.audit_backends import azure_monitor
from agent_os.audit_backends.azure_monitor import AzureMonitorBackend
from agent_os.audit_logger import AuditEntry, GovernanceAuditLogger


class FakeCredential:
    """Fake Azure credential for backend tests."""


class FakeLogsIngestionClient:
    """Fake Azure Monitor client for backend tests."""

    def __init__(self, endpoint: str, credential: Any) -> None:
        self.endpoint = endpoint
        self.credential = credential
        self.upload_calls: list[dict[str, Any]] = []
        self.closed = False
        self.raise_error = False

    def upload(self, *, rule_id: str, stream_name: str, logs: list[dict[str, Any]]) -> None:
        if self.raise_error:
            raise RuntimeError("upload failed")
        self.upload_calls.append(
            {
                "rule_id": rule_id,
                "stream_name": stream_name,
                "logs": logs,
            }
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the backend module with a fake Azure Monitor SDK."""
    monkeypatch.setattr(azure_monitor, "_HAS_AZURE_MONITOR", True)
    monkeypatch.setattr(azure_monitor, "_LogsIngestionClient", FakeLogsIngestionClient)
    monkeypatch.setattr(azure_monitor, "_DefaultAzureCredential", FakeCredential)


def make_entry(index: int = 0, **overrides: Any) -> AuditEntry:
    """Create a test audit entry."""
    entry = AuditEntry(
        timestamp="2025-01-01T00:00:00+00:00",
        event_type="governance_decision",
        agent_id=f"agent-{index}",
        action=f"action-{index}",
        decision="allow",
        reason="policy_ok",
        latency_ms=12.5,
        metadata={"index": index},
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def test_graceful_degradation_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """write and flush are safe no-ops when the SDK is unavailable."""
    monkeypatch.setattr(azure_monitor, "_HAS_AZURE_MONITOR", False)
    monkeypatch.setattr(azure_monitor, "_LogsIngestionClient", None)
    monkeypatch.setattr(azure_monitor, "_DefaultAzureCredential", None)

    backend = AzureMonitorBackend("https://example.com", "dcr-id", "Custom-Stream")

    backend.write(make_entry())
    backend.flush()
    backend.close()

    assert backend.enabled is False
    assert backend._buffer == []


def test_enabled_false_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """enabled is false when Azure Monitor dependencies are missing."""
    monkeypatch.setattr(azure_monitor, "_HAS_AZURE_MONITOR", False)

    backend = AzureMonitorBackend("https://example.com", "dcr-id", "Custom-Stream")

    assert backend.enabled is False


def test_write_buffers_entry(fake_sdk: None) -> None:
    """write buffers entries until the batch threshold is reached."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )

    backend.write(make_entry())

    assert len(backend._buffer) == 1
    assert backend._client.upload_calls == []


def test_flush_sends_buffered_entries(fake_sdk: None) -> None:
    """flush uploads the buffered records."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )
    backend.write(make_entry())

    backend.flush()

    assert len(backend._client.upload_calls) == 1
    assert backend._client.upload_calls[0]["rule_id"] == "dcr-id"
    assert backend._client.upload_calls[0]["stream_name"] == "Custom-Stream"
    assert backend._client.upload_calls[0]["logs"][0]["AgentId"] == "agent-0"
    assert backend._buffer == []


def test_auto_flush_on_batch_size(fake_sdk: None) -> None:
    """write auto-flushes when the batch size is reached."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=2,
    )

    backend.write(make_entry(1))
    backend.write(make_entry(2))

    assert len(backend._client.upload_calls) == 1
    assert len(backend._client.upload_calls[0]["logs"]) == 2
    assert backend._buffer == []


def test_flush_error_does_not_crash(fake_sdk: None, caplog: pytest.LogCaptureFixture) -> None:
    """flush logs upload errors without raising them."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )
    backend.write(make_entry())
    backend._client.raise_error = True

    with caplog.at_level(logging.ERROR, logger=azure_monitor.__name__):
        backend.flush()

    assert "Failed to upload governance audit entries to Azure Monitor" in caplog.text


def test_thread_safety(fake_sdk: None) -> None:
    """Concurrent writes preserve all buffered records."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=1000,
    )

    def writer(start: int) -> None:
        for index in range(start, start + 25):
            backend.write(make_entry(index, action=f"action-{index}"))

    threads = [threading.Thread(target=writer, args=(offset * 25,)) for offset in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    actions = {record["Action"] for record in backend._buffer}
    assert len(backend._buffer) == 500
    assert actions == {f"action-{index}" for index in range(500)}


def test_close_flushes_remaining(fake_sdk: None) -> None:
    """close flushes buffered records and closes the client."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )
    backend.write(make_entry())

    backend.close()

    assert len(backend._client.upload_calls) == 1
    assert backend._client.closed is True


def test_entry_dict_format(fake_sdk: None) -> None:
    """Uploaded records use the Azure Monitor flat schema."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )
    backend.write(
        make_entry(
            7,
            timestamp="2025-02-03T04:05:06+00:00",
            event_type="policy_violation",
            agent_id="agent-7",
            action="tool_call",
            decision="deny",
            reason="blocked",
            latency_ms=99.9,
            metadata={"tool": "shell", "count": 3},
        )
    )

    backend.flush()

    payload = backend._client.upload_calls[0]["logs"][0]
    assert payload == {
        "TimeGenerated": "2025-02-03T04:05:06+00:00",
        "EventType": "policy_violation",
        "AgentId": "agent-7",
        "Action": "tool_call",
        "Decision": "deny",
        "Reason": "blocked",
        "LatencyMs": 99.9,
        "Metadata": json.dumps({"tool": "shell", "count": 3}, default=str),
    }


def test_integration_with_governance_audit_logger(fake_sdk: None) -> None:
    """GovernanceAuditLogger writes entries through the backend."""
    backend = AzureMonitorBackend(
        "https://example.com",
        "dcr-id",
        "Custom-Stream",
        batch_size=10,
    )
    audit_logger = GovernanceAuditLogger()
    audit_logger.add_backend(backend)

    audit_logger.log_decision(
        agent_id="agent-42",
        action="search",
        decision="allow",
        reason="within_policy",
        latency_ms=4.2,
        source="unit-test",
    )
    audit_logger.flush()

    payload = backend._client.upload_calls[0]["logs"][0]
    assert payload["EventType"] == "governance_decision"
    assert payload["AgentId"] == "agent-42"
    assert payload["Action"] == "search"
    assert payload["Decision"] == "allow"
    assert json.loads(payload["Metadata"]) == {"source": "unit-test"}
