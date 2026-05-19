import sys
from pathlib import Path
curr_dir = Path.cwd()
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))

from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState
from langchain_openai import ChatOpenAI
from Financagent.Phrase_3_MultiAgent.core.tools import detect_anomalies
from Financagent.Phrase_3_MultiAgent.core.memory_manager import update_tool_history
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [detect_anomalies]

agent = create_agent(model=llm,
                     tools=tools,
                     system_prompt=f"""
                                    You are a financial fraud and anomaly detection specialist.
                                    Your job is to identify unusual, suspicious, or unexpected transactions.
                                    Use detect_anomalies to find transactions that deviate significantly from historical averages.
                                    When reporting anomalies, always include the transaction description, amount, and how far above average it is.
                                    If no anomalies are found, say so clearly — do not fabricate suspicious activity.
                                """)

def anomaly_node(state: FinanceSystemState):
    feedback = state.revision_feedback
    revision_context = ""

    if state.needs_revision:
        revision_context = f"""
                    REVISION REQUESTED:
                    Your previous response was inadequate. Specific feedback:
                    {feedback}
                    Previous response: {state.agent_outputs.get('analyst', '')}
                    Please address all issues in your new response.
                    """

    messages = list(state.messages)
    if revision_context:
        messages.append(revision_context)

    response = agent.invoke({'messages': messages})

    new_message = response['messages'][len(state.messages):]

    final_response = response['messages'][-1].content

    updated_tool_history = update_tool_history(state.tool_history, new_message)

    return {
        'agent_outputs': {'anomaly': final_response},
        'tool_history': updated_tool_history
    }