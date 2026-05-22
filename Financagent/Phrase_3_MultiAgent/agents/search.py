import sys
from pathlib import Path
curr_dir = Path(__file__).parent.parent
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))

from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Financagent.Phrase_3_MultiAgent.core.tools import search_transactions
from Financagent.Phrase_3_MultiAgent.core.memory_manager import update_tool_history
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [search_transactions]

agent = create_agent(model = llm,
                      tools = tools,
                      system_prompt=f"""
                                    You are a transaction search specialist.
                                    Your job is to find specific transactions matching the user's description.
                                    Use search_transactions with the most relevant keywords from the user's question.
                                    Always include the date, description, and amount for each result.
                                    If no matching transactions are found, say so clearly.
                                """ )

def search_node(state: FinanceSystemState):

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
        'agent_outputs': {'search': final_response},
        'tool_history': updated_tool_history
    }