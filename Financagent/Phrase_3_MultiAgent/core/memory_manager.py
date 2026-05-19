from pydantic import BaseModel
from typing import List, Optional
import datetime as dt
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from .state import FinanceSystemState

class Entity(BaseModel):
    name: str
    note: str

class ExtractContent(BaseModel):
    year: Optional[int]
    month: Optional[str]
    category: Optional[str]
    last_question_type: Optional[str]
    new_entity: List[Entity]


CATEGORIES = ["Food & Dining", "Groceries", "Gas", "Subscription", "Entertainment", "Technology", "Alcohol", "Smoke",
              "Payment", "Shopping", "Transportation", "Shipping", "Charity"]
QUESTION_TYPE = ['Spending_Summary', 'Trend', 'Anomaly', 'Search']

llm = ChatOpenAI(model = 'gpt-4o-mini')
structured_llm = llm.with_structured_output(ExtractContent)

def extract_content(message, curr_state, llm):
    system_prompt = f""" You are a highly precise financial data assistant. Your job is to analyze the user's latest message, maintain the state of their current query context, and determine their analytical intent.

                        ### Current Context & Inputs:
                        - **Today's date : {dt.datetime.today().strftime('%Y-%m-%d')}. 
                        - **Available Category: {CATEGORIES}
                        - **Current State: {curr_state}
                        - **User lastest message

                        ### ⚠️ HIGHEST PRIORITY RULE — Read Before All Others:
                        - NEVER use today's date as a default for current_year or current_month.
                        - Today's date is provided for relative time resolution ONLY (e.g. "last month", "this year").
                        - If the user's message contains NO explicit or relative time reference,
                        - AND the current state has no active year/month,
                        - return current_year = null and current_month = null.
                        - No exceptions.

                        ### Strict State-Keeping Rules:
                        1. **Inheritance:** If the user's message does NOT mention a new time period, retain the existing state value IF one exists. If existing state is null, return null — do not substitute today's date.
                        2. **Overrides:** If the user explicitly mentions a new Year, Month, or Category, update that specific field with the new value. 
                        3. **Relative Time Resolution:** - "This month" refers to the month of {dt.datetime.now().month}.
                                                        - "Last year" refers to the year prior to {dt.datetime.now().year}.
                                                        - If a specific month is mentioned without a year (e.g., "in October"), default to the current state's year, or the most recent past occurrence of that month relative to today.
                                                        - If NO month is mentioned at all, return None for current_month — do not infer or default to the current month. Preserve the existing state month only if it is directly relevant to the follow-up question.
                                                        - If the user asks about a full year with no specific month mentioned, set current_month to None regardless of existing state. A question like "in 2024" or "for the year 2025" always means year-level query — never infer a month for these.

                        ### Intent Classification (`last_question_type`):
                        Classify the user's latest message into exactly one of these four categories:
                        - `spending_summary`: User wants to know total spend, averages, or general breakdowns for a period (e.g., "How much did I spend in total?", "Show me my dinner expenses").
                        - `trend`: User is asking about changes, trajectories, or comparisons over time (e.g., "Am I spending more than last month?", "Show me my monthly trend for groceries").
                        - `anomaly`: User is looking for unusual activity, spikes, duplicates, or suspicious charges (e.g., "Are there any weird charges?", "Did I get double-billed?").
                        - `search`: User is looking for specific individual transactions or list of items (e.g., "Find my receipts from Amazon", "Show me transactions over $100").     

                        ### New Entities Identification (`new_entities`):
                        Extract any specific vendor names, specific items, or custom flags the user mentions. 
                        - Format this as a JSON object of key-value pairs where the key is the entity and the value is a short note/context (e.g., `{{"VIRTUALINSTRUCT": "user flagged as suspicious"}}`). 
                        - If no new entities are mentioned, return an empty object ``. 

                    """

    user_prompt = f"Please process the following query based on the CURRENT STATE {curr_state} and the lastest user message {message}"

    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]

    response = llm.invoke(messages)

    return response


def update_tool_history(tool_history, new_messages):
    """After each agent turn, scan the new messages for tool calls and their results."""
    new_entries = []

    for message in new_messages:
        if isinstance(message, AIMessage):
            tool_calls = message.tool_calls
            for tool_call in tool_calls:
                selected_tool = tool_call['name']
                args = tool_call['args']
                new_entries.append({
                    'tools': selected_tool,
                    'args': args,
                    'result_summary': None,
                })

        elif isinstance(message, ToolMessage):
            for entry in reversed(new_entries):
                if entry['result_summary'] is None:
                    result_summary = message.content[:100]
                    entry['result_summary'] = result_summary

    tool_history.extend(new_entries)

    return tool_history[-20:]


def update_summary(summary, messages, iteration, llm):
    if iteration % 10 != 0:
        return summary

    system_prompt = SystemMessage(
        f'You are a very professional summarizer that could summarize old summary as well as new message and generate a new string of summary.')
    user_prompt = HumanMessage(
        f"""Please summarize this for me, here is the existing summary {summary} and this is the recent message {messages}.""")
    prompt = [system_prompt, user_prompt]
    response = llm.invoke(prompt)
    return response.content


def build_contextualized_message(user_input, state):
    """Take the raw user input and current state, generate a single enriched string."""

    if not any([state.current_year, state.current_month,
                state.current_category, state.entities,
                state.tool_history]):
        return user_input

    # Determine query level explicitly
    if state.current_year and not state.current_month:
        query_level = f"YEAR-LEVEL QUERY for {state.current_year} — call tool with year parameter only, do NOT ask for a month"
    elif state.current_month:
        query_level = f"MONTH-LEVEL QUERY for {state.current_month}"
    else:
        query_level = "NO TIME PERIOD — ask user for clarification before calling any tool"

    # Only inject non-None fields
    context_parts = [f"QUERY LEVEL: {query_level}"]
    if state.current_year:
        context_parts.append(f"Year: {state.current_year}")
    if state.current_month:
        context_parts.append(f"Month: {state.current_month}")
    if state.current_category:
        context_parts.append(f"Category: {state.current_category}")
    if state.last_question_type:
        context_parts.append(f"Question type: {state.last_question_type}")
    if state.entities:
        context_parts.append(f"Known entities: {state.entities}")
    if state.tool_history:
        context_parts.append(f"Recent tools used: {state.tool_history[-3:]}")

    context_block = "\n                ".join(context_parts)

    return f"""
                [CONTEXT]
                {context_block}
                [/CONTEXT]

                User: {user_input}
            """

def has_time_reference(message: str) -> bool:
    """Check if message contains any explicit or relative time reference."""

    time_keywords = [
        "this month", "last month", "this year", "last year",
        "today", "currently", "now", "recent"
    ]

    # Check for relative keywords
    if any(kw in message.lower() for kw in time_keywords):
        return True

    # Check for explicit year (4-digit number)
    if re.search(r'\b20\d{2}\b', message):
        return True

    # Check for explicit month names
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    if any(m in message.lower() for m in months):
        return True

    # Check for YYYY-MM format
    if re.search(r'\b20\d{2}-\d{2}\b', message):
        return True

    return False

def memory_manager_node(state: FinanceSystemState):

    # Extract the lastest message
    latest_message = state.messages[-1]

    # Purpose of debugging
    # print(f"\n[DEBUG] Latest message: {latest_message.content}")
    # print(f"[DEBUG] Has time reference: {has_time_reference(latest_message.content)}")
    # print(f"[DEBUG] Current state year: {state.current_year}")
    # print(f"[DEBUG] Current state month: {state.current_month}")

    extracted_content = extract_content(latest_message, state, structured_llm)

    # print(f"[DEBUG] Extracted year: {extracted_content.year}")
    # print(f"[DEBUG] Extracted month: {extracted_content.month}")

    if not has_time_reference(latest_message.content):
        if state.current_year is None:
            extracted_content.year = None
        if state.current_month is None:
            extracted_content.month = None

    # Update the entities
    new_entities_dict = {e.name: e.note for e in extracted_content.new_entity}
    merged_dict = state.entities | new_entities_dict

    # calculate turn count
    turn_count = len([m for m in state.messages if isinstance(m, HumanMessage)])

    # summarize the conversation if turn exceed 10 turns
    summary = update_summary(state.summary, state.messages, turn_count, llm)

    # enrich the raw user input and replace it with the old message
    removed_message = RemoveMessage(id = latest_message.id)
    enriched_input = HumanMessage(build_contextualized_message(latest_message.content, state))

    return {
        'messages': [removed_message, enriched_input],
        'current_year': extracted_content.year,
        'current_month': extracted_content.month,
        'current_category': extracted_content.category,
        'last_question_type': extracted_content.last_question_type,
        'entities': merged_dict,
        'summary': summary,

        # Reset every Turn
        'agent_outputs': {},
        'final_response': None,
        'needs_report': False,
        'active_agents': [],
        'revision_count': 0
    }