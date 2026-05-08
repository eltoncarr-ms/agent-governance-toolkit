# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for create_audit_logger factory function."""

from __future__ import annotations

import builtins
import os
from unittest.mock import patch

import pytest

from agent_os.audit_logger import (
    GovernanceAuditLogger,
    InMemoryBackend,
    JsonlFileBackend,
    LoggingBackend,
    _register_builtin_backends,
    create_audit_logger,
)


class TestCreateAuditLogger:
    """Tests for the audit logger factory."""

    def test_default_uses_logging_backend(self):
        """No args and no env var → LoggingBackend."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIT_BACKENDS", None)
            audit = create_audit_logger()
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], LoggingBackend)

    def test_explicit_logging(self):
        audit = create_audit_logger("logging")
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], LoggingBackend)

    def test_explicit_memory(self):
        audit = create_audit_logger("memory")
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], InMemoryBackend)

    def test_jsonl_with_path(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        audit = create_audit_logger(f"jsonl:path={path}")
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], JsonlFileBackend)
        assert str(audit._backends[0].path) == path
        audit._backends[0].close()

    def test_multiple_backends(self, tmp_path):
        path = str(tmp_path / "audit.jsonl")
        audit = create_audit_logger(f"jsonl:path={path},logging")
        assert len(audit._backends) == 2
        assert isinstance(audit._backends[0], JsonlFileBackend)
        assert isinstance(audit._backends[1], LoggingBackend)
        audit._backends[0].close()

    def test_env_var_used_when_no_arg(self):
        with patch.dict(os.environ, {"AUDIT_BACKENDS": "memory"}):
            audit = create_audit_logger()
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], InMemoryBackend)

    def test_explicit_arg_overrides_env_var(self):
        with patch.dict(os.environ, {"AUDIT_BACKENDS": "memory"}):
            audit = create_audit_logger("logging")
        assert len(audit._backends) == 1
        assert isinstance(audit._backends[0], LoggingBackend)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown audit backend 'nonexistent'"):
            create_audit_logger("nonexistent")

    def test_unknown_backend_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            create_audit_logger("bogus")

    def test_otel_backend_import_error_message(self):
        """otel backend gives helpful error when SDK missing."""
        with patch.dict("agent_os.audit_logger._BACKEND_REGISTRY", {}, clear=True):
            _register_builtin_backends()

        real_import = builtins.__import__

        def _raising_import(name, *args, **kwargs):
            if name == "agent_os.otel_audit_backend":
                raise ImportError("missing otel")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_raising_import):
            with pytest.raises(ImportError, match="OTel audit backend requires opentelemetry-sdk"):
                create_audit_logger("otel")

    def test_empty_spec_returns_empty_logger(self):
        audit = create_audit_logger("")
        assert len(audit._backends) == 0

    def test_whitespace_handling(self):
        audit = create_audit_logger("  logging , memory  ")
        assert len(audit._backends) == 2
        assert isinstance(audit._backends[0], LoggingBackend)
        assert isinstance(audit._backends[1], InMemoryBackend)

    def test_returns_governance_audit_logger_type(self):
        audit = create_audit_logger("logging")
        assert isinstance(audit, GovernanceAuditLogger)
