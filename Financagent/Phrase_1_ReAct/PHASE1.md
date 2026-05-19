# Personal Finance AI Agent — Phase 1 Documentation

## Overview

A conversational AI agent that ingests personal bank transaction data, categorizes spending using an LLM, indexes transactions for semantic search, and answers natural language questions about spending patterns, trends, and anomalies. Built from scratch using LangGraph, LangChain, ChromaDB, and OpenAI.

---

## Problem Statement

Managing personal finances requires manually reviewing bank statements to track spending patterns, detect unusual charges, and compare trends over time. There was no conversational, intelligent way to query transaction history in plain English without building it yourself.

---

## Architecture Overview

```
Raw Bank CSV
     ↓
┌─────────────────┐
│ Ingestion Layer │  ← ingestion.py
│  - Clean & normalize    │
│  - LLM categorization   │
└────────┬────────┘
         ↓
┌─────────────────┐
│   RAG Layer     │  ← rag.py
│  - ChromaDB index       │
│  - Semantic search      │
└────────┬────────┘
         ↓
┌─────────────────────────────────┐
│         Agent Layer             │  ← agent.py
│                                 │
│  ┌──────────────────────────┐   │
│  │   Memory Manager Node    │   │
│  │  - Context extraction    │   │
│  │  - State management      │   │
│  │  - Context injection     │   │
│  └────────────┬─────────────┘   │
│               ↓                 │
│  ┌──────────────────────────┐   │
│  │      Agent Node          │   │
│  │  - Tool routing          │   │
│  │  - Response generation   │   │
│  └────────────┬─────────────┘   │
│               ↓                 │
│         Tools Layer             │  ← tools.py
│  - get_monthly_spending_summary │
│  - get_monthly_comparison       │
│  - detect_anomalies             │
│  - search_transactions          │
└─────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT-4o-mini |
| Vector Store | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Data Processing | Pandas |
| Language | Python 3.12 |
| Testing | Pytest |
| Environment | Conda |

---

## Project Structure

```
finance_agent/
├── data/
│   ├── raw/                    ← raw bank CSV exports
│   └── processed/              ← normalized transaction data
├── chroma_db/                  ← ChromaDB persistent store
├── test/
│   ├── __init__.py
│   └── e2e.py                  ← end-to-end test suite
├── ingestion.py                ← data ingestion and categorization
├── rag.py                      ← vector store indexing and querying
├── tools.py                    ← agent tools
├── agent.py                    ← memory manager, agent node, graph
├── .env                        ← API keys
└── requirements.txt
```

---

## Components

### 1. Ingestion Layer — `ingestion.py`

Transforms raw bank CSV exports into a clean, enriched dataset ready for the agent.

**Input Schema (raw bank CSV):**
| Column | Description |
|---|---|
| DATE | Transaction date |
| DESCRIPTION | Merchant description |
| AMOUNT | Negative for debits, positive for credits |
| CHECK # | Check number (nullable) |
| STATUS | Posted or Pending |

**Output Schema (normalized CSV):**
| Column | Description |
|---|---|
| DATE | Parsed datetime |
| DESCRIPTION | Original merchant description |
| AMOUNT | Float, signed |
| IS_DEBIT | Boolean derived from AMOUNT |
| CATEGORY | LLM-inferred category label |
| MONTH | YYYY-MM string |
| YEAR | Integer year |
| CHECK_NUMBER | Renamed, nullable |
| STATUS | Filtered to Posted only |

**Categorization Pipeline:**
- Descriptions batched in groups of 20 to minimize API calls
- GPT-4o-mini with `with_structured_output()` enforces valid category labels
- Memory-backed retry loop: on failure, appends bad response + correction message to conversation before retrying
- Fallback to `"Other"` after 3 failed attempts

**Available Categories:**
```
Food & Dining, Groceries, Gas, Subscription, Entertainment,
Technology, Alcohol, Smoke, Payment, Shopping, Transportation,
Shipping, Charity, Other
```

---

### 2. RAG Layer — `rag.py`

Indexes all transactions into ChromaDB for semantic search alongside metadata filtering.

**Document Format:**
Each transaction is converted to a natural language string:
```
$45.23 debit at STARBUCKS #1234 on 2024-03-12, category: Food & Dining
```

**Metadata Fields:**
```python
{
    "amount": float,
    "date": str,
    "month": str,       # YYYY-MM
    "year": int,
    "category": str,    # lowercase
    "is_debit": str,    # "true" / "false" (ChromaDB doesn't support booleans)
}
```

**Key Design Decisions:**
- Document text drives semantic similarity search
- Metadata enables exact filtering (month, category, is_debit)
- Collection reset on re-indexing prevents duplicate ID conflicts

---

### 3. Tools Layer — `tools.py`

Four `@tool` decorated functions the agent can call based on user intent.

#### `get_monthly_spending_summary(month=None, category=None, year=None)`
Returns spending totals and category breakdowns for a given month or year.
- Month and year are both optional — year alone triggers year-level aggregation
- Returns pre-aggregated category totals to minimize LLM synthesis burden

#### `get_monthly_comparison(category=None, year=None)`
Compares spending month-over-month across transaction history.
- Optional category filter scopes to a single spending type
- Optional year filter scopes to a single year

#### `detect_anomalies(month=None, year=None)`
Flags transactions more than 2 standard deviations above their category's historical average.
- Calculates per-category mean and std across all historical data
- Returns flagged transactions with amount and distance above average

#### `search_transactions(question, month=None, category=None, year=None)`
Semantically searches transaction history using a natural language question.
- Combines ChromaDB semantic search with metadata filtering
- Returns formatted transaction list with metadata

---

### 4. Agent Layer — `agent.py`

A LangGraph stateful graph with two nodes: Memory Manager and Agent.

#### Shared State — `AgentState`

```python
@dataclass
class AgentState:
    messages: Annotated[list, add_messages]  # full conversation history
    current_year: Optional[int] = None       # active year context
    current_month: Optional[str] = None      # active month context  
    current_category: Optional[str] = None   # active category context
    last_question_type: Optional[str] = None # spending_summary | trend | anomaly | search
    summary: Optional[str] = None            # rolling conversation summary
    entities: dict = field(default_factory=dict)      # vendor/category memory
    tool_history: list = field(default_factory=list)  # recent tool calls
```

#### Memory Manager Node

Runs before the agent every turn. Responsibilities:

1. **Context Extraction** — calls GPT-4o-mini with structured output to extract year, month, category, question type, and new entities from the latest user message. Preserves existing state when nothing new is mentioned.

2. **Query Level Classification** — programmatically determines query intent:
   - `year set, month None` → YEAR-LEVEL QUERY
   - `month set` → MONTH-LEVEL QUERY  
   - `both None` → NO TIME PERIOD — ask user

3. **Context Injection** — replaces the raw user message with an enriched version containing current state, query level, entities, and recent tool history.

4. **Summary Update** — regenerates conversation summary every 10 turns to prevent context window bloat.

5. **Tool History Update** — tracks recent tool calls and their results for context.

#### Agent Node

Invokes the ReAct agent with the enriched messages and updates tool history from the response.

#### Graph Definition

```
START → memory_manager_node → agent_node → END
```

Compiled with `MemorySaver` for cross-turn persistence.

---

## Key Technical Challenges

### 1. LLM Categorization Reliability
**Problem:** Batched categorization returned inconsistent counts — extra or missing labels corrupted the entire batch mapping.

**Solution:** Switched from positional list matching to structured output with `dict[str, str]` mapping (description → category), making each label lookup independent of position. Used `with_structured_output()` with a Pydantic model to enforce schema compliance.

### 2. Context Retention Across Turns
**Problem:** Agent repeatedly asked for clarification on follow-up questions like "How about 2025?" even when year context was established.

**Solution:** Built a Memory Manager node that maintains structured state and injects explicit query level labels (`YEAR-LEVEL QUERY for 2025 — do NOT ask for a month`) directly into each message. Prompt engineering alone was insufficient — deterministic state management was required.

### 3. Agent Defaulting to Current Date
**Problem:** When no time period was specified, the agent defaulted to today's date and returned $0 spending for the current month.

**Root Cause:** Tool functions had `month` as a required parameter (no default), forcing the agent to supply a value.

**Solution:** Made all time parameters optional (`month=None`, `year=None`) and added explicit guard clauses in each tool.

---

## Test Suite

End-to-end tests covering three behavioral dimensions:

### `TestContextRetention`
| Test | Validates |
|---|---|
| `test_year_followup` | Agent carries year context to follow-up questions |
| `test_category_followup_after_month` | Agent retains month when asking about categories |
| `test_yes_confirmation` | Agent doesn't reset on single-word confirmations |
| `test_month_without_year` | Agent inherits year when only month is mentioned |

### `TestToolRouting`
| Test | Validates |
|---|---|
| `test_routes_to_spending_summary` | Clear spending questions hit correct tool |
| `test_routes_to_anomaly_detection` | Anomaly questions routed correctly |
| `test_routes_to_trend` | Trend questions return multi-month data |
| `test_routes_to_search` | Vendor searches use semantic search tool |

### `TestClarificationBehavior`
| Test | Validates |
|---|---|
| `test_no_clarification_on_year_followup` | No redundant clarification on year follow-ups |
| `test_no_clarification_on_category_followup` | No month re-asking after category questions |
| `test_no_clarification_on_month_followup` | Month follow-ups inherit year from context |
| `test_no_clarification_on_yes_confirmation` | "Yes" responses don't reset context |
| `test_first_turn_clarification_is_acceptable` | Clarification acceptable with zero context |

---

## Running the Project

### Setup
```bash
conda create -n ai_agent python=3.12
conda activate ai_agent
pip install -r requirements.txt
```

### Environment Variables
```bash
# .env
OPENAI_API_KEY=your_key_here
```

### Run Ingestion
```bash
python ingestion.py --input data/raw/transactions.csv --output data/processed/normalized.csv
```

### Index Transactions
```bash
python rag.py
```

### Run Agent
```bash
python agent.py
```

### Run Tests
```bash
pytest test/e2e.py -v
```

---

## Sample Interactions

```
User: How much did I spend in 2024?
Agent: In 2024, your total spending was $4,709.98. Here's the monthly breakdown...

User: How about 2025?
Agent: In 2025, your total spending was $7,138.15. Here's the monthly breakdown...

User: Which category was highest?
Agent: In 2025, Groceries was your highest spending category at $1,788.01...

User: Anything unusual in August 2025?
Agent: I detected 3 unusual transactions in August 2025:
       - OTHER charge: $1,599.05 (exceeds average by $986.05)
       ...
```

---

## Lessons Learned

1. **Prompt engineering has limits** — deterministic Python guardrails (required vs optional parameters, explicit state injection) are more reliable than LLM instruction for enforcing structured behavior.

2. **Pre-aggregate in tools, not in the LLM** — returning raw month-by-month data and asking the LLM to synthesize it is slower, more expensive, and less reliable than returning pre-computed aggregates.

3. **Structured output over JSON prompting** — `with_structured_output()` eliminates an entire class of parsing failures that retry loops only partially address.

4. **Test what the agent should NOT do** — `assert_no_clarification()` and negative assertions caught regressions that positive assertions missed entirely.

---

## Phase 2 — Planned

- Multi-source ingestion (normalize multiple bank and credit card CSV formats)
- Cross-account spending analysis
- Income tracking and net cash flow reporting

## Phase 3 — Planned

Multi-agent architecture with specialized agents:
- **Orchestrator Agent** — routes queries and coordinates
- **Ingestion Agent** — pulls and normalizes data
- **Analysis Agent** — answers spending questions
- **Anomaly Agent** — monitors for unusual patterns
- **Report Writer Agent** — generates narrative summaries

---

*Built as part of a self-directed AI agent development curriculum.*
