# Governance Middleware Architecture

Agent OS provides a multi-layer governance middleware stack for the
Microsoft Agent Framework (MAF). Each layer enforces a different aspect
of governance and can be composed independently.

## Stack Composition

When all layers are enabled the stack is ordered as follows:

| # | Middleware | Type | Scope |
|---|-----------|------|-------|
| 1 | `AuditTrailMiddleware` | AgentMiddleware | Tamper-proof audit logging |
| 2 | `GovernancePolicyMiddleware` | AgentMiddleware | Declarative policy enforcement (pre-LLM) |
| 3 | `GovernedToolMiddleware` | FunctionMiddleware | Policy-evaluated tool gating (pre-tool) |
| 4 | `CapabilityGuardMiddleware` | FunctionMiddleware | Static tool allow/deny lists |
| 5 | `RogueDetectionMiddleware` | FunctionMiddleware | Behavioral anomaly detection |

**Layer 2** evaluates *message-level* rules before the LLM responds.
**Layer 3** evaluates *tool-level* rules after the LLM requests a tool
but before the tool executes. Both layers use the same `PolicyEvaluator`
and can share YAML policy files.

## Quick Start

```python
from agent_framework import Agent
from agent_os.integrations.maf_adapter import create_governance_middleware

stack = create_governance_middleware(
    policy_directory="policies/",
    allowed_tools=["web_search", "read_file"],
    denied_tools=["delete_file"],
    enable_governed_tool_guard=True,   # enables Layer 3
    agent_id="my-researcher",
)

agent = Agent(name="researcher", middleware=stack)
```

## MCP Tool Governance

Remote MCP tools bypass the `FunctionMiddleware` pipeline entirely —
they are resolved server-side by the chat client. To govern MCP tools,
use the companion factory:

```python
from agent_os.integrations.maf_adapter import (
    create_governance_middleware,
    create_mcp_governance_gateway,
)

# Local tool governance (middleware stack)
middleware = create_governance_middleware(
    policy_directory="policies/",
    denied_tools=["execute_code"],
)

# MCP tool governance (gateway)
mcp_gateway = create_mcp_governance_gateway(
    denied_tools=["execute_code"],
    audit_log=my_audit_log,  # shared with middleware for unified audit
)

# Before each MCP tool call:
allowed, reason = mcp_gateway.intercept_tool_call(
    agent_id="my-agent",
    tool_name="execute_code",
    params={"code": "print('hello')"},
)
if not allowed:
    # Return denial to LLM as tool output
    return f"⛔ Tool blocked: {reason}"
```

## Policy Files

A single YAML policy file can contain rules for both layers. The
evaluator only matches rules whose `field` appears in the evaluation
context, so message rules never fire during tool evaluation and vice
versa.

See [`examples/policies/tool_guard_policy.yaml`](../examples/policies/tool_guard_policy.yaml)
for a complete example.

### Message-level context (Layer 2)

| Field | Type | Description |
|-------|------|-------------|
| `agent` | str | Agent name |
| `message` | str | Last user message text |
| `timestamp` | float | Current time |
| `message_count` | int | Number of messages |

### Tool-level context (Layer 3)

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Tool being invoked |
| `tool_args` | str | Flattened arguments (for regex matching) |
| `tool_args_structured` | dict | Structured arguments |
| `agent_id` | str | Agent identity |

## Enable/Disable Strategy

| Scenario | Result |
|----------|--------|
| `enable_governed_tool_guard=True` | GovernedToolMiddleware added to stack |
| `enable_governed_tool_guard=False` (default) | Not added — backward compatible |
| No `policy_directory` | Neither policy middleware nor tool guard |
