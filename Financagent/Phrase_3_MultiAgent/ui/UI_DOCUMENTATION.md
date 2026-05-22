# Financagent UI Documentation

## Overview

The Financagent UI is a Streamlit web interface for the Phase 3 multi-agent personal finance system. It expands the original command-line experience into a simple browser-based application with three main views:

- Chat: ask natural language finance questions.
- Dashboard: explore spending patterns from the processed transaction dataset.
- Annual Report: generate a yearly AI-powered financial report.

The UI keeps the existing LangGraph multi-agent backend intact. User requests are sent to `Phrase_3_MultiAgent.main.run()`, which invokes the compiled graph and returns both the assistant response and selected trace details for display.

## Technology Stack

| Layer | Technology |
|---|---|
| UI framework | Streamlit |
| Charts | Plotly Express, Plotly Graph Objects |
| Data handling | Pandas |
| Agent backend | LangGraph, LangChain |
| LLM integration | LangChain OpenAI |
| Dataset | `data/processed/categorized_data.csv` |

## File Structure

```text
Phrase_3_MultiAgent/
├── main.py                  # Backend run() function used by the UI
├── graph.py                 # LangGraph multi-agent workflow
├── agents/                  # Analyst, anomaly, search, critic, report writer, orchestrator
├── core/                    # Shared state, tools, memory manager
└── ui/
    ├── app.py               # Streamlit app entry point and sidebar navigation
    ├── chat.py              # Chat screen and agent trace display
    ├── dashboard.py         # Spending dashboard and transaction explorer
    ├── report.py            # Annual report generation screen
    └── UI_DOCUMENTATION.md  # This documentation
```

## How To Run

From the project root:

```bash
streamlit run Phrase_3_MultiAgent/ui/app.py
```

The app will open in the browser at Streamlit's local URL, usually:

```text
http://localhost:8501
```

Before running the UI, make sure the required environment variables are available through `.env`, especially the OpenAI API key used by the LangChain/OpenAI backend.

## Main Entry Point

The UI starts from:

```text
Phrase_3_MultiAgent/ui/app.py
```

This file is responsible for:

- configuring the Streamlit page;
- initializing session state;
- loading basic dataset summary metrics;
- rendering the sidebar;
- routing between Chat, Dashboard, and Annual Report pages.

## Session State

The app uses `st.session_state` to preserve UI state across Streamlit reruns.

| Key | Purpose |
|---|---|
| `thread_id` | Unique conversation ID passed into the LangGraph checkpointer |
| `messages` | Chat history shown in the Chat tab |
| `agent_traces` | Reserved for trace history |
| `show_traces` | Controls whether the agent reasoning path is shown |

When the user clicks `New Conversation`, the UI creates a new `thread_id`, clears messages, clears traces, and reruns the app.

## Backend Contract

The UI calls:

```python
run(user_input: str, thread_id: str)
```

The current backend returns a dictionary:

```python
{
    "response": "...",
    "routed_to": [...],
    "critic_scores": {...},
    "revision_count": 0,
    "needs_revision": False,
    "tool_history": [...]
}
```

The Chat and Report pages expect `result["response"]` to contain the final assistant-facing answer.

## Sidebar

The sidebar appears on every page and includes:

- `New Conversation` button
- shortened thread ID
- dataset metrics
- agent reasoning path toggle
- page navigation
- current chat message count

Dataset metrics are loaded from:

```text
data/processed/categorized_data.csv
```

The sidebar currently displays:

- total transactions
- total debit spending
- dataset date range

## Chat Page

Implemented in:

```text
Phrase_3_MultiAgent/ui/chat.py
```

The Chat page allows users to ask natural language questions such as:

- "How much did I spend in 2025?"
- "Find my Costco charges."
- "Anything unusual last month?"
- "How much did I spend and was any of it unusual?"

Flow:

1. User enters a question in `st.chat_input`.
2. The message is appended to `st.session_state.messages`.
3. The UI calls `run(user_input, thread_id)`.
4. The backend routes the question through the Phase 3 LangGraph system.
5. The assistant response is rendered in the chat.
6. The response and trace metadata are saved into session state.

If `Show agent reasoning path` is enabled, the assistant response includes an expandable trace panel.

## Agent Trace Panel

The trace panel displays selected internal execution details:

- agents routed to by the orchestrator;
- critic feedback or revision information;
- revision count;
- recent tool calls.

This is useful for demonstrating the multi-agent architecture without exposing the full LangGraph state object.

## Dashboard Page

Implemented in:

```text
Phrase_3_MultiAgent/ui/dashboard.py
```

The Dashboard page provides an interactive financial overview using the processed transaction data.

Current features:

- year selector;
- month selector;
- total spending metric;
- transaction count metric;
- largest transaction metric;
- top spending category metric;
- spending by category horizontal bar chart;
- spending distribution donut chart;
- monthly spending trend line chart;
- searchable transaction table;
- category filter for the transaction table.

The dashboard filters to debit transactions:

```python
df[df["IS_DEBIT"] == True]
```

Transaction amounts are converted to positive values using `abs()` before visualization.

## Annual Report Page

Implemented in:

```text
Phrase_3_MultiAgent/ui/report.py
```

The Annual Report page lets the user select a year and generate a full AI financial report.

Flow:

1. User selects a year.
2. User clicks `Generate Report`.
3. The UI sends this prompt to the backend:

```text
Generate a full annual report for {selected_year}
```

4. The multi-agent backend routes the request through the report workflow.
5. The report is displayed as Markdown.
6. The user can download the report as a Markdown text file.

## Data Requirements

The UI assumes the processed dataset exists at:

```text
data/processed/categorized_data.csv
```

Expected columns include:

| Column | Used By |
|---|---|
| `DATE` | dashboard date parsing and table |
| `DESCRIPTION` | transaction search field |
| `AMOUNT` | spending metrics and charts |
| `CATEGORY` | category charts and filters |
| `MONTH` | month selector and trend chart |
| `YEAR` | year selector and report selector |
| `IS_DEBIT` | debit-only spending filter |

## User Workflow

Typical user flow:

1. Open the app with Streamlit.
2. Review dataset information in the sidebar.
3. Use Chat to ask personalized finance questions.
4. Enable agent traces when debugging or demonstrating the multi-agent system.
5. Use Dashboard to inspect spending visually.
6. Use Annual Report to generate and download a year-level summary.

## Developer Notes

### Import Path Handling

The UI files manually update `sys.path` so they can import the project package from the local workspace. This works for local development, but a cleaner long-term approach is to package the repository or run the app as an installed module.

### Streamlit Reruns

Streamlit reruns the script after interactions. Persistent state should be stored in `st.session_state`, not normal Python variables.

### Caching

Dataset loading uses `@st.cache_data` to avoid reloading CSV files on every rerun. If the CSV changes while the app is running, clear the Streamlit cache or restart the app.

### Conversation Memory

The Chat page keeps a stable `thread_id` for the current conversation. This allows LangGraph's checkpointer to preserve conversation context across multiple user questions.

The Annual Report page intentionally uses a new random thread ID for each generated report so report generation is isolated from the active chat conversation.

## Current Limitations

- There is no authentication or multi-user account system.
- Chat history is stored only in the current Streamlit session.
- Reports are downloaded as Markdown text, not PDF or DOCX.
- The agent trace panel shows a compact subset of backend state, not the full graph execution.
- The dashboard depends on the processed CSV schema remaining stable.

## Future Improvements

Possible next steps:

- Add persistent conversation history.
- Add PDF export for annual reports.
- Add filters for merchant, amount range, and date range.
- Add anomaly markers directly on dashboard charts.
- Add a dedicated "Agent Trace" page for deeper debugging.
- Replace manual `sys.path` edits with a proper package/module setup.
- Add user-uploaded CSV support.
- Add tests for UI data loading and backend response shape.

