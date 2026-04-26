# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for BaseIntegration.from_policy_file() factory."""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_os.integrations.base import (
    BaseIntegration, GovernancePolicy, GovernanceEventType,
)
from agent_os.integrations.langchain_adapter import LangChainKernel


# Concrete test kernel for non-LangChain tests
class _TestKernel(BaseIntegration):
    def wrap(self, agent):
        return agent
    def unwrap(self, agent):
        return agent


@pytest.fixture
def policy_file(tmp_path):
    """Create a temp policy YAML file."""
    policy = GovernancePolicy(
        name="test-policy",
        max_tokens=8192,
        max_tool_calls=20,
        timeout_seconds=120,
    )
    path = tmp_path / "policy.yaml"
    policy.save(str(path))
    return path


@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "audit.jsonl"


class TestTypeCorrectness:
    def test_langchain_returns_correct_type(self, policy_file):
        kernel = LangChainKernel.from_policy_file(str(policy_file))
        assert isinstance(kernel, LangChainKernel)

    def test_test_kernel_returns_correct_type(self, policy_file):
        kernel = _TestKernel.from_policy_file(str(policy_file))
        assert isinstance(kernel, _TestKernel)


class TestPolicyLoading:
    def test_policy_matches_yaml(self, policy_file):
        kernel = LangChainKernel.from_policy_file(str(policy_file))
        assert kernel.policy.name == "test-policy"
        assert kernel.policy.max_tokens == 8192
        assert kernel.policy.max_tool_calls == 20

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LangChainKernel.from_policy_file(str(tmp_path / "nonexistent.yaml"))

    def test_malformed_yaml_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: [valid: yaml: {{", encoding="utf-8")
        with pytest.raises(Exception):  # Could be ValueError or yaml error
            LangChainKernel.from_policy_file(str(bad_file))


class TestLangChainTimeout:
    def test_timeout_auto_aligned(self, policy_file):
        kernel = LangChainKernel.from_policy_file(str(policy_file))
        assert kernel.timeout_seconds == 120  # From policy

    def test_timeout_explicit_override(self, policy_file):
        kernel = LangChainKernel.from_policy_file(
            str(policy_file), timeout_seconds=60
        )
        assert kernel.timeout_seconds == 60


class TestAuditWiring:
    def test_audit_with_path(self, policy_file, audit_path):
        kernel = LangChainKernel.from_policy_file(
            str(policy_file), audit_path=str(audit_path)
        )
        assert kernel.audit_logger is not None
        assert hasattr(kernel.audit_logger, 'log_decision')
        assert hasattr(kernel.audit_logger, 'flush')

    def test_governance_events_produce_audit(self, policy_file, audit_path):
        kernel = LangChainKernel.from_policy_file(
            str(policy_file), audit_path=str(audit_path)
        )
        # Emit a governance event
        kernel.emit(GovernanceEventType.POLICY_CHECK, {
            "agent_id": "test", "phase": "pre_execute"
        })
        kernel.audit_logger.flush()
        # Read the JSONL file
        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["event_type"] == "governance_decision"

    def test_no_audit_when_omitted(self, policy_file):
        kernel = LangChainKernel.from_policy_file(str(policy_file))
        assert kernel.audit_logger is None


class TestNoDoubleIO:
    def test_policy_loaded_once(self, policy_file):
        """LangChain override should not cause double file reads."""
        import unittest.mock as mock
        original_from_yaml = GovernancePolicy.from_yaml

        call_count = 0
        @classmethod
        def counting_from_yaml(cls, yaml_str):
            nonlocal call_count
            call_count += 1
            return original_from_yaml.__func__(cls, yaml_str)

        with mock.patch.object(GovernancePolicy, 'from_yaml', counting_from_yaml):
            LangChainKernel.from_policy_file(str(policy_file))

        # Should be called exactly once (in LangChain override), not twice
        assert call_count == 1, f"from_yaml called {call_count} times, expected 1"


class TestBackwardsCompatibility:
    def test_direct_constructor_still_works(self):
        """Existing LangChainKernel(policy=...) usage unchanged."""
        policy = GovernancePolicy()
        kernel = LangChainKernel(policy=policy)
        assert kernel.policy is policy

    def test_test_kernel_direct_constructor(self):
        kernel = _TestKernel()
        assert isinstance(kernel.policy, GovernancePolicy)
