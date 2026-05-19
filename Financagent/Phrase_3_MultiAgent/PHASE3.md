# Personal Finance AI Agent — Phase 3 Documentation
## Multi-Agent System with Self-Reflection

---

## Overview

Phase 3 expands the single-agent architecture from Phase 1 into a fully orchestrated multi-agent system. Specialized agents run in parallel, a critic evaluates output quality and triggers revision loops, and an orchestrator synthesizes results into coherent responses. The system is built on LangGraph's stateful graph with shared state, parallel dispatch, and conditional routing.

---

## Problem Statement

A single ReAct agent handling all financial query types becomes a generalist — adequate at everything but excellent at nothing. At company scale, this creates bottlenecks, makes maintenance difficult, and prevents specialization. Phase 3 introduces dedicated agents for each analytical domain, enabling parallel execution and quality-controlled outputs.

---

## Architecture Overview

```
START
  ↓
memory_manager          ← context extraction, state enrichment
  ↓
orchestrator_routing    ← decides which agents to invoke
  ↓ (parallel dispatch)
┌──────────┬──────────┬────────┐
analyst   anomaly   search    ← run simultaneously
└──────────┴──────────┴────────┘
  ↓ (fan-in)
critic                 ← evaluates output quality
  ↓ (conditional)
needs_revision → retry → [analyst, anomaly, search] → critic  ← loop (max 3)
no revision    → orchestrator_synthesis or report_writer
  ↓
END
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| Primary LLM | OpenAI GPT-4o (orchestrator, critic, report writer) |
| Agent LLM | OpenAI GPT-4o-mini (analyst, anomaly, search) |
| Memory LLM | OpenAI GPT-4o-mini (memory manager, context extraction) |
| Vector Store | ChromaDB (reused from Phase 1) |
| Embeddings | OpenAI text-embedding-3-small |
| Data Processing | Pandas |
| Language | Python 3.12 |

---

## Project Structure

```
phase3/
├── core/
│   ├── __init__.py
│   ├── state.py            ← FinanceSystemState shared across all agents
│   ├── tools.py            ← all tools + tool group constants
│   └── memory_manager.py   ← context extraction, state enrichment
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py     ← routing, synthesis, conditional edge functions
│   ├── analyst.py          ← spending and trend agent
│   ├── anomaly.py          ← anomaly detection agent
│   ├── search.py           ← semantic transaction search agent
│   ├── report_writer.py    ← narrative report generation
│   └── critic.py           ← quality evaluation and revision control
├── graph.py                ← full graph definition and compilation
├── main.py                 ← CLI entry point
└── test/
    └── e2e.py
```

---

## Shared State — `state.py`

The foundation of the multi-agent system. All agents read from and write to this shared state.

```python
@dataclass
class FinanceSystemState:
    # Conversation memory
    messages: Annotated[list, add_messages]

    # Context (set by memory manager)
    current_year: Optional[int] = None
    current_month: Optional[str] = None
    current_category: Optional[str] = None
    last_question_type: Optional[str] = None
    summary: Optional[str] = None
    entities: dict = field(default_factory=dict)
    tool_history: Annotated[list, extend_list] = field(default_factory=list)

    # Multi-agent coordination (set by orchestrator)
    routing_decision: list = field(default_factory=list)
    active_agents: list = field(default_factory=list)
    needs_report: bool = False

    # Agent outputs (set by each agent, merged by reducer)
    agent_outputs: Annotated[dict, merge_dicts] = field(default_factory=dict)

    # Critic control
    needs_revision: bool = False
    revision_count: int = 0
    revision_feedback: Annotated[dict, merge_dicts] = field(default_factory=dict)

    # Final output
    final_response: Optional[str] = None
```

### Reducers

| Field | Reducer | Purpose |
|---|---|---|
| `messages` | `add_messages` | Appends new messages, prevents overwrites |
| `agent_outputs` | `merge_dicts` | Merges parallel agent results safely |
| `tool_history` | `extend_list` | Concatenates and trims to last 10 entries |
| `revision_feedback` | `merge_dicts` | Merges parallel critic feedback |

---

## Components

### 1. Memory Manager — `core/memory_manager.py`

Identical to Phase 1. Runs before every agent turn to extract context and enrich messages.

Additional Phase 3 responsibility — resets coordination fields each turn:
```python
'agent_outputs': {},
'revision_count': 0,
'needs_revision': False,
'revision_feedback': {},
'active_agents': [],
'final_response': None,
```

Without this reset, revision counts and stale agent outputs would carry over between user turns.

---

### 2. Tools — `core/tools.py`

All Phase 1 tools reused unchanged, plus one new tool:

#### `get_full_annual_report(year)` ← New in Phase 3
Used exclusively by the Report Writer agent. Returns a comprehensive data dump:
- Total annual spending
- Category breakdown aggregated across whole year
- Month-by-month totals
- Top 5 largest individual transactions
- Months containing anomalous transactions

**Tool Groups:**
```python
ANALYST_TOOLS = [get_monthly_spending_summary, get_monthly_comparison]
ANOMALY_TOOLS = [detect_anomalies]
SEARCH_TOOLS = [search_transactions]
REPORT_TOOLS = [get_full_annual_report]
```

---

### 3. Orchestrator — `agents/orchestrator.py`

Two node functions with distinct responsibilities:

#### `orchestrator_routing_node`
Routes the query to the appropriate agents using structured output:

```python
class RoutingDecision(BaseModel):
    agents: list[str]    # e.g. ["analyst", "anomaly"]
    needs_report: bool   # True only for explicit report requests
    reasoning: str       # brief explanation
```

**Routing Rules:**
| Question Type | Agents Invoked |
|---|---|
| Spending totals, breakdowns, trends | `["analyst"]` |
| Unusual or suspicious charges | `["anomaly"]` |
| Specific vendor or transaction lookup | `["search"]` |
| Compound multi-intent question | `["analyst", "anomaly"]` etc. |
| Explicit report request | `["analyst", "anomaly", "search"]` + `needs_report=True` |

#### `orchestrator_synthesis_node`
Runs after all agents complete (when `needs_report=False`). Reads `agent_outputs` and synthesizes a single coherent response. Clearly labels each agent's contribution in the prompt to avoid data repetition.

#### `route_to_agents(state)`
Conditional edge function. Returns `state.routing_decision` — LangGraph uses this to dispatch agents in parallel.

#### `route_after_critic(state)`
```python
def route_after_critic(state) -> str:
    if state.needs_revision and state.revision_count < 3:
        return 'retry'
    elif state.needs_report:
        return 'report_writer'
    else:
        return 'orchestrator_synthesis'
```

The `revision_count < 3` guard prevents infinite loops.

---

### 4. Specialized Agents

Each follows the same pattern: receive state, invoke with tools, return output and updated tool history.

#### `analyst_node` — `agents/analyst.py`
- **Tools:** `get_monthly_spending_summary`, `get_monthly_comparison`
- **Specialization:** Spending totals, category breakdowns, month-over-month trends
- **Returns:** `{'agent_outputs': {'analyst': response}}`

#### `anomaly_node` — `agents/anomaly.py`
- **Tools:** `detect_anomalies`
- **Specialization:** Identifying transactions deviating from historical averages
- **Returns:** `{'agent_outputs': {'anomaly': response}}`

#### `search_node` — `agents/search.py`
- **Tools:** `search_transactions`
- **Specialization:** Semantic transaction lookup by vendor or description
- **Returns:** `{'agent_outputs': {'search': response}}`

#### Revision Feedback Handling
Each agent checks for revision feedback before invoking:

```python
feedback = state.revision_feedback.get('analyst')
if feedback:
    messages.append(HumanMessage(f"""
        REVISION REQUESTED:
        Your previous response was inadequate.
        Feedback: {feedback['feedback']}
        Issues found: {feedback['issues']}
        Previous response: {state.agent_outputs.get('analyst', '')}
        Please address all issues in your new response.
    """))
```

---

### 5. Critic Agent — `agents/critic.py`

The quality control layer. Evaluates each active agent's output against a strict rubric and determines whether revision is needed.

#### Structured Output
```python
class CriticAgentFeedback(BaseModel):
    score: int           # 1-5 (threshold: 4)
    issue: list[str]     # specific problems found
    feedback: str        # actionable revision instruction

class CriticDecision(BaseModel):
    analyst: Optional[CriticAgentFeedback]
    anomaly: Optional[CriticAgentFeedback]
    search: Optional[CriticAgentFeedback]
    needs_revision: bool
```

#### Quality Rubrics

**Analyst must:**
- Contain specific $ amounts
- Cover the full time period requested
- Include category breakdown when asked about spending
- Be directly responsive to the user's question

**Anomaly must:**
- List specific transaction descriptions and amounts
- Explain WHY each transaction is anomalous
- Cover the time period requested
- Explicitly state if no anomalies found

**Search must:**
- Return actual matching transactions with dates and amounts
- Be specific to what was searched for
- Explicitly state if no matches found

#### Revision Decision
`needs_revision` is derived programmatically from scores rather than trusting the LLM's own assessment:

```python
needs_revision = any([
    decision.analyst and decision.analyst.score < 4,
    decision.anomaly and decision.anomaly.score < 4,
    decision.search and decision.search.score < 4,
])
```

---

### 6. Report Writer — `agents/report_writer.py`

Pure synthesis agent — no tools, no tool-calling. Calls `get_full_annual_report()` directly and generates a structured narrative.

**Report Structure:**
1. Executive Summary
2. Spending by Category
3. Monthly Trends
4. Notable Transactions
5. Anomalies & Unusual Activity
6. Actionable Observations (exactly 3, data-driven)

Only runs when `needs_report=True`. Sets `final_response` directly — `orchestrator_synthesis` is skipped.

---

## Graph Definition — `graph.py`

```python
graph = StateGraph(FinanceSystemState)

# Nodes
graph.add_node('memory_manager', memory_manager_node)
graph.add_node('orchestrator_routing', orchestrator_routing_node)
graph.add_node('analyst', analyst_node)
graph.add_node('anomaly', anomaly_node)
graph.add_node('search', search_node)
graph.add_node('critic', critic_node)
graph.add_node('retry', retry_node)
graph.add_node('orchestrator_synthesis', orchestrator_synthesis_node)
graph.add_node('report_writer', report_writer_node)

# Entry flow
graph.add_edge(START, 'memory_manager')
graph.add_edge('memory_manager', 'orchestrator_routing')

# Parallel dispatch
graph.add_conditional_edges('orchestrator_routing', route_to_agents,
    ['analyst', 'anomaly', 'search'])

# All agents → critic
graph.add_edge('analyst', 'critic')
graph.add_edge('anomaly', 'critic')
graph.add_edge('search', 'critic')

# Critic → conditional routing
graph.add_conditional_edges('critic', route_after_critic, {
    'retry': 'retry',
    'report_writer': 'report_writer',
    'orchestrator_synthesis': 'orchestrator_synthesis'
})

# Retry → back to agents (parallel)
graph.add_conditional_edges('retry', route_to_agents,
    ['analyst', 'anomaly', 'search'])

# Terminal nodes
graph.add_edge('orchestrator_synthesis', END)
graph.add_edge('report_writer', END)

app = graph.compile(checkpointer=MemorySaver())
```

---

## Key Technical Challenges

### 1. Parallel State Updates Without Race Conditions
**Problem:** Multiple agents running simultaneously both try to update `tool_history` and `agent_outputs`, causing `InvalidUpdateError`.

**Solution:** Added reducers to all fields updated by parallel agents:
- `agent_outputs` → `merge_dicts` reducer
- `tool_history` → `extend_list` reducer
- `revision_feedback` → `merge_dicts` reducer

### 2. Revision Count Carrying Across Turns
**Problem:** `revision_count` accumulated across conversation turns, meaning by turn 4 the critic's retry loop was permanently disabled.

**Solution:** Reset all coordination fields in `memory_manager_node` at the start of each turn.

### 3. Critic Over-Trusting LLM's Self-Assessment
**Problem:** The LLM's `needs_revision` field in structured output was inconsistent — sometimes flagging revision when all scores were 4+.

**Solution:** Derived `needs_revision` programmatically from individual agent scores rather than relying on the LLM's boolean judgment.

### 4. Agent Statelessness in Parallel Execution
**Problem:** Parallel agents cannot safely write to shared state simultaneously.

**Solution:** Each agent returns only its own output key (`{'agent_outputs': {'analyst': ...}}`). LangGraph's `merge_dicts` reducer safely merges all parallel outputs after execution completes.

---

## Sample Interactions

```
Ask: How much did I spend in 2025?
[CRITIC] analyst: 4/5 — passed
Agent: In 2025, your total spending was $8,139.15...

Ask: Find my Hilton charges
[CRITIC] search: 5/5 — passed
Agent: Hilton charges:
       - $1,011.08 at Hilton Memphis (Aug 9)
       - $267.39 at Hilton Tapestry Atlanta (Aug 14)

Ask: How much did I spend and was anything unusual in 2025?
[CRITIC] analyst: 5/5, anomaly: 4/5 — both passed
Agent: Total spending $8,139.15... August flagged with $1,751.01 ACH payment...

Ask: Generate a full annual report for 2025
[CRITIC] all agents: 4/5 — passed
Agent: [Full structured annual report with 6 sections]
```

---

## Design Decisions

### Why Stateless Individual Agents?
Parallel agents read from shared state but never write to it during execution. They return outputs which LangGraph merges after all complete. This prevents race conditions and makes the system deterministic regardless of execution order.

### Why a Dedicated Critic Rather Than Self-Reflection?
Self-reflection (agent evaluating its own output) has an inherent bias — agents tend to approve their own work. A separate critic with an explicit rubric provides objective quality control and makes evaluation criteria transparent and adjustable.

### Why `needs_revision` Derived Programmatically?
LLM boolean judgments are inconsistent. Deriving `needs_revision` from score thresholds is deterministic — the same rubric produces the same decision every time, making the system behavior predictable and debuggable.

### Why Report Writer Has No Tools?
The report writer always performs the same operation: call `get_full_annual_report()` and generate a narrative. Tool-calling capability adds complexity without benefit when the action is fixed. A direct function call is simpler, faster, and more reliable.

---

## Running the System

### Setup
```bash
conda activate ai_agent
cd phase3
```

### Run Agent
```bash
python main.py
```

### Example Queries
```
How much did I spend in 2025?
Anything unusual in August 2025?
Find my Amazon charges
How much did I spend and was anything unusual in 2025?
Generate a full annual report for 2025
```

---

## Comparison: Phase 1 vs Phase 3

| Dimension | Phase 1 | Phase 3 |
|---|---|---|
| Architecture | Single ReAct agent | Multi-agent orchestrated system |
| Execution | Sequential tool calls | Parallel agent dispatch |
| Specialization | Generalist | Domain-specific agents |
| Quality Control | None | Critic with revision loop |
| Report Generation | Not supported | Dedicated report writer |
| State Management | Simple AgentState | Rich FinanceSystemState with reducers |
| Routing | Agent decides tool | Orchestrator routes to agents |
| Self-improvement | None | Critic-driven revision (max 3 iterations) |

---

## Phase 4 — Planned

- Multi-tenant support (multiple users, multiple accounts)
- Real-time data ingestion via bank API webhooks
- Scheduled weekly email reports
- Web dashboard with interactive charts
- Cross-account analysis and net worth tracking

---

*Built as part of a self-directed AI agent development curriculum.*
*Phase 1 documentation available in PHASE1.md*
