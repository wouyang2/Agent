from pathlib import Path
import sys

curr_dir = Path(__file__).parent.parent
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))
from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

class RoutingDecision(BaseModel):
    agents: list[str] = None
    needs_report: bool = False
    reasoning: str = None

llm = ChatOpenAI(model= 'gpt-4o')
structured_llm = llm.with_structured_output(RoutingDecision)

AVAILABLE_AGENTS = ['analyst', 'anomaly', 'search']

def orchestrator_routing_node(state: FinanceSystemState = None):

    # Read the enriched message from manager
    enriched_messages = state.messages
    question_type = state.last_question_type

    question_type_str = question_type if question_type else "Not yet classified — infer from the user's message directly"
    year_str = str(state.current_year) if state.current_year else "Not specified"
    month_str = state.current_month if state.current_month else "Not specified"

    # call llm with routing prompt
    system_prompt = SystemMessage(f"""You are an orchestrator for a personal finance AI system.
    Your job is to analyze the user's question and decide which specialized agents to invoke.

    AVAILABLE AGENTS AND THEIR SPECIALIZATIONS:
    - analyst: Handles spending totals, category breakdowns, month-over-month trends, 
               and annual summaries. Use for questions about "how much", "breakdown", "trend".
    - anomaly: Detects unusual or suspicious transactions by comparing against historical 
               averages. Use for questions about "unusual", "suspicious", "weird", "spike".
    - search: Finds specific transactions or vendors using semantic search. 
              Use for questions about specific merchants, keywords, or individual charges.

    ROUTING RULES:
    - Single intent question → route to ONE agent only
    - Compound question with multiple intents → route to ALL relevant agents
    - Explicit report request ("generate a report", "full analysis", "summarize everything") 
      → route to all agents AND set needs_report = True
    - needs_report = True ONLY for explicit report requests, NOT for regular questions
    - "Find my Costco charges" → agents: ["search"] ONLY
      → Do NOT add anomaly unless the user explicitly asks about unusual activity.
      → Vendor lookup questions are ALWAYS search only.

    CURRENT CONTEXT:
    - Question type classified as: {question_type_str}
    - Active year: {year_str}
    - Active month: {month_str}

    ROUTING EXAMPLES:
    - "How much did I spend in 2025?" → agents: ["analyst"], needs_report: False
    - "Anything unusual last month?" → agents: ["anomaly"], needs_report: False
    - "Find my Amazon charges" → agents: ["search"], needs_report: False
    - "How much did I spend and was any of it unusual?" → agents: ["analyst", "anomaly"], needs_report: False
    - "Generate a full annual report for 2025" → agents: ["analyst", "anomaly", "search"], needs_report: True

    Your reasoning should be brief and explain why you chose each agent.
    """)

    user_prompt = HumanMessage(f"Route this question to the appropriate agents: {enriched_messages[-1].content}")

    messages = [system_prompt, user_prompt]

    response = structured_llm.invoke(messages)

    return {
        'routing_decision': response.agents,
        'active_agents': response.agents,
        'needs_report': response.needs_report,
    }

def orchestrator_synthesis_node (state: FinanceSystemState = None):
    """This node runs after all parallel agents complete."""

    # Read agent output from state
    agent_outputs = state.agent_outputs

    # Read the original user query
    enriched_messages = state.messages

    system_prompt = SystemMessage(f"""You are a financial report synthesizer. 
    Your job is to combine outputs from multiple specialized agents into a single 
    clear, coherent response for the user.

    RULES:
    - Present a unified response — do not repeat the same data multiple times
    - If multiple agents returned data, weave them together naturally
    - Always present amounts as positive numbers with a $ sign
    - Do not mention agent names or internal system details in your response
    - If an agent returned no results, acknowledge it briefly and move on
    - Keep the response concise but complete
    - Use bullet points and headers only when the response is complex enough to warrant it

    AVAILABLE AGENT OUTPUTS:
    {chr(10).join(f"[{agent.upper()}]: {output}" for agent, output in agent_outputs.items())}
    """)

    user_prompt = HumanMessage(f"""The user asked: {enriched_messages[-1].content}

    Please synthesize the agent outputs above into a single coherent response 
    that directly answers the user's question.""")

    messages = [system_prompt, user_prompt]
    response = llm.invoke(messages)

    return {
        'final_response': response.content,
        'messages': [AIMessage(response.content)]
    }

def route_to_agents(state: FinanceSystemState) -> list[str]:
    """Called by LangGraph conditional edges to determine which agents to invoke."""
    return state.routing_decision

def route_after_agents(state: FinanceSystemState) -> str:
    if state.needs_report:
        return 'report_writer'
    else:
        return 'orchestrator_synthesis'

def route_after_critic(state: FinanceSystemState):
    if state.needs_revision and state.revision_count < 3:
        return 'retry'
    if state.needs_report:
        return 'report_writer'
    else:
        return 'orchestrator_synthesis'


if __name__ == "__main__":

    mock_state = FinanceSystemState(
        messages=[HumanMessage("How much did I spend in 2025?")],
    )
    result = orchestrator_routing_node(mock_state)
    print(result)