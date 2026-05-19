import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from tools import get_monthly_comparison, get_monthly_spending_summary, detect_anomalies, search_transactions
from helper import extract_content, update_summary, build_contextualized_message, update_tool_history, has_time_reference

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage
from langgraph.graph import START, END, StateGraph

import datetime as dt
import uuid
from typing import Annotated, Optional, List, Dict
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

@dataclass
class AgentState:
    messages: Annotated[list, add_messages]
    current_year: Optional[int] = None
    current_month: Optional[str] = None
    current_category: Optional[str] = None
    last_question_type: Optional[str] = None
    summary: Optional[str] = None
    entities: dict = field(default_factory=dict)
    tool_history: list = field(default_factory=list)

class Entity(BaseModel):
    name: str
    note: str

class ExtractContent(BaseModel):
    year: Optional[int]
    month: Optional[str]
    category: Optional[str]
    last_question_type: Optional[str]
    new_entity: List[Entity]

helper_llm = ChatOpenAI(model = 'gpt-4o-mini')
structured_helper_llm = helper_llm.with_structured_output(ExtractContent)

CATEGORIES = ["Food & Dining", "Groceries", "Gas", "Subscription", "Entertainment", "Technology", "Alcohol", "Smoke", "Payment", "Shopping", "Transportation", "Shipping", "Charity"]
llm = ChatOpenAI(model = 'gpt-4o-mini')

# Creating Agent
agent = create_agent(
    model = llm,
    tools = [get_monthly_comparison, detect_anomalies, search_transactions, get_monthly_spending_summary],
    system_prompt=f"""You are a personal finance analyst with access to the user's complete 
    transaction history through a set of tools. Your job is to answer questions accurately 
    — never guess or fabricate transaction data. If a tool returns no results, say so clearly.

    TODAY'S DATE: {dt.date.today()}

    AVAILABLE CATEGORIES:
    {', '.join(CATEGORIES)}

    TOOL ROUTING GUIDE:
    - User asks about totals, expenses, or category breakdowns → get_monthly_spending_summary
    - User asks about specific vendors, keywords, or charges → search_transactions
    - User asks about suspicious, unusual, or unexpected charges → detect_anomalies
    - User asks about trends or month-over-month changes → get_monthly_comparison
    
    CRITICAL TIME PERIOD RULE:
    - When context shows current_year = null AND current_month = null, do NOT pass any month or year to any tool — ask the user first.
    - When current_year is provided but current_month is null, this means a YEAR-LEVEL query. Use the year parameter only — do NOT ask for a month. Year alone is always sufficient.
    - When current_month is provided, use it directly.
    - NEVER use today's date to fill missing parameters.
    - Defaulting to a time period not specified by the user is strictly forbidden.

    DATE HANDLING RULES:
    - Months are always passed as YYYY-MM format (e.g. '2024-03')
    - Years are always passed as integers (e.g. 2024), not strings
    - If the user says 'last month', calculate it from today's date
    - If the user says 'this year' or 'last year', calculate accordingly
    - When a year is mentioned without a month, use the year parameter — do not loop through months

    BEHAVIOR RULES:
    - When the user's intent is clear from conversation history, act immediately — do not ask for clarification
    - Only ask for clarification when the question is genuinely ambiguous with no context to infer from
    - If a question spans multiple tools, call them in sequence and synthesize the results
    - Always present amounts as positive numbers with a $ sign in your response
    
    CONTEXT INFERENCE RULES:
    - Track the most recently mentioned year and month throughout the conversation
    - If the user asks "how about [year]?" assume the same question type as before
    - If the user asks "how about [month]?" assume the same year as the last answer
    - If the user answers "yes" to a clarifying question, execute what was proposed — do not restart
    - Only forget context when the user explicitly starts a new topic
    - When in doubt, state your assumption and proceed rather than asking
      e.g. "Assuming you mean January 2026 based on our conversation..."
    - When the user asks a follow-up question without specifying a time period, always use the most recently discussed month or year — never ask again.
      Example: if the last answer was about May 2024, and the user asks "what category did I spend the most on?", assume May 2024 immediately.
    - Asking for clarification about time period is ONLY acceptable when no month or year has been mentioned anywhere in the conversation so far. If any time period exists in conversation history, use it.
    """
)

def memory_manager_node(state: AgentState):

    # Extract the lastest message
    latest_message = state.messages[-1]

    print(f"\n[DEBUG] Latest message: {latest_message.content}")
    print(f"[DEBUG] Has time reference: {has_time_reference(latest_message.content)}")
    print(f"[DEBUG] Current state year: {state.current_year}")
    print(f"[DEBUG] Current state month: {state.current_month}")

    extracted_content = extract_content(latest_message, state, structured_helper_llm)

    print(f"[DEBUG] Extracted year: {extracted_content.year}")
    print(f"[DEBUG] Extracted month: {extracted_content.month}")

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
    summary = update_summary(state.summary, state.messages, turn_count, helper_llm)

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
    }

def agent_node(state: AgentState):

    response = agent.invoke({'messages': state.messages})

    new_message = response['messages'][len(state.messages) - 1:]

    updated_tool_history = update_tool_history(state.tool_history,new_message)

    return {
        'messages': new_message,
        'tool_history': updated_tool_history,
    }

def run_agent(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    response = app.invoke(
        {"messages": [HumanMessage(user_input)]},
        config=config
    )

    return response["messages"][-1].content


graph = StateGraph(AgentState)
graph.add_node('memory_manager', memory_manager_node)
graph.add_node('agent', agent_node)

graph.add_edge(START, 'memory_manager')
graph.add_edge('memory_manager', 'agent')
graph.add_edge('agent', END)

memory = MemorySaver()

app = graph.compile(checkpointer=memory)

if __name__ == '__main__':
    thread_id = str(uuid.uuid4())

    while True:

        userInput = input('Ask: ')

        if userInput.lower() in ['quit', 'exit']:
            print("Exiting...")
            break

        response = run_agent(userInput, thread_id)

        print(response)

