# Module 08 — Model Context Protocol (MCP)

**Time:** 3–5 days · **Depends on:** 07 · **Next:** [Advanced RAG](09-advanced-rag.md)

!!! important "Correct definition (industry)"
    **MCP** (Model Context Protocol) is an open standard—originating from Anthropic and now broadly adopted—for connecting AI applications (hosts/clients) to external **tools**, **resources**, and **prompts** via MCP servers.

    It is **not** a multi-model load balancer. Routing and load balancing belong in [Production](13-production.md) and [Integration patterns](16-integration-patterns.md). Earlier drafts of this course mislabeled those topics as MCP; that is corrected here.

---

## Learning objectives

- Explain hosts, clients, and servers in MCP
- Know tools vs resources vs prompts
- Reason about security when installing MCP servers

## What you can build

- An assistant that reads local project files through an MCP filesystem server
- A custom MCP server exposing a small internal API as tools
- Policy for which servers are allowed in dev vs prod

---

## Architecture

```text
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│ Host app     │────▶│ MCP Client  │────▶│ MCP Server(s)    │
│ (IDE, chat,  │     │ (session,   │     │ tools/resources  │
│  agent)      │     │  sampling)  │     │ prompts          │
└──────────────┘     └─────────────┘     └──────────────────┘
```

| Role | Responsibility |
|------|----------------|
| **Host** | UX, auth, orchestration (Claude Desktop, VS Code, your agent) |
| **Client** | Protocol session with servers |
| **Server** | Exposes capabilities over stdio or HTTP/SSE transports |

### Capability types

| Type | Use |
|------|-----|
| **Tools** | Side-effecting or computed actions (`search_issues`, `run_query`) |
| **Resources** | Readable data blobs (`file://`, `db://schema`) |
| **Prompts** | Reusable prompt templates distributed by the server |

---

## Why MCP exists

Before MCP, every IDE/agent reinvented N integrations. MCP standardizes:

- Discovery of tools/resources  
- Schema for tool arguments  
- A path to consistent auth and auditing in the host  

Your app becomes **model-agnostic at the tool boundary**: swap models without rewriting every connector.

---

## Security model (non-negotiable)

MCP servers can be as powerful as local code execution.

| Risk | Control |
|------|---------|
| Malicious server | Install only reviewed sources; pin versions |
| Over-broad filesystem | Scope roots; read-only where possible |
| Secret exfiltration | Host approval for tool calls; redact logs |
| Confused deputy | Per-tenant credentials; never share user tokens blindly |
| Supply chain | Lockfiles; SBOM; internal registry for prod |

**Human-in-the-loop** for destructive tools (delete, pay, email send).

---

## Conceptual custom tool server

Pseudocode — use the official MCP SDKs for Python/TypeScript in real projects:

```python
# Conceptual — follow current MCP SDK docs for real servers
"""
Server name: acme-tickets
Tools:
  - list_tickets(status: open|closed) -> list[Ticket]
  - get_ticket(id: str) -> Ticket
Resources:
  - ticket://{id}
"""

def list_tickets(status: str = "open") -> list[dict]:
    # query internal API with service credentials
    return [{"id": "T-1", "title": "Login timeout", "status": status}]
```

Hosts list tools → model selects tool + args → host/client invokes server → result returns to model context.

---

## Relation to Module 07 tools

| Module 07 tools | MCP tools |
|-----------------|-----------|
| In-process Python functions | Out-of-process, reusable across hosts |
| App-specific | Shareable across IDE, desktop, agents |
| You own the loop | Host may own sampling + UI approvals |

Use **in-process tools** for simple apps; use **MCP** when you want portable connectors or IDE integration.

---

## Multi-model routing (not MCP)

Still important — just correctly named:

```python
def route_model(task: str) -> str:
    if task in {"classify", "extract"}:
        return "gpt-4o-mini"  # or local SLM
    if task == "deep_reason":
        return "claude-sonnet-4-20250514"
    return "gpt-4o"
```

See Module 10 (cost) and 13 (production) for caching, fallbacks, and load shedding.

---

## Exercise

1. Install a reputable MCP filesystem or git server in a **dev-only** host.  
2. List tools/resources; perform one read-only operation.  
3. Write a policy doc: which servers are allowed in CI vs laptop vs prod.

---

## Checkpoint

- [ ] You can define MCP without saying “load balancer”  
- [ ] You know tools vs resources  
- [ ] You have a security bar for third-party servers  

**Next:** [Module 09 — Advanced RAG](09-advanced-rag.md)

### Primary references

- [Model Context Protocol specification](https://modelcontextprotocol.io/)  
- Anthropic / ecosystem MCP server registries (verify before install)
