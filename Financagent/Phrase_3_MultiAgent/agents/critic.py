from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
curr_dir = Path.cwd()
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))

from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

class CriticAgentFeedback(BaseModel):
    score: int = 0   # 1-5
    issue: list[str] # Problem found
    feedback: str    # Suggestion on improvement

class CriticDecision(BaseModel):
    analyst: Optional[CriticAgentFeedback]  # None if analyst wasn't invoked
    anomaly: Optional[CriticAgentFeedback]  # None if anomaly wasn't invoked
    search: Optional[CriticAgentFeedback]  # None if search wasn't invoked
    needs_revision: bool  # True if any agent scored below threshold


llm = ChatOpenAI(model='gpt-4o')
structured_llm = llm.with_structured_output(CriticDecision)

def critic_node(state: FinanceSystemState):

    if not state.agent_outputs:
        return {'needs_revision': False}

    original_question = state.messages[-1].content

    system_prompt = SystemMessage(f"""You are a quality control critic for a personal finance AI system.
                                    Your job is to evaluate each agent's output against a strict quality rubric.

                                    QUALITY RUBRICS:

                                    ANALYST output must:
                                    - Contain specific $ amounts (not just categories)
                                    - Cover the full time period requested
                                    - Include category breakdown when asked about spending
                                    - Be directly responsive to the user's question
                                    Score below 4 if ANY of these are missing.

                                    ANOMALY output must:
                                    - List specific transaction descriptions and amounts
                                    - Explain WHY each transaction is anomalous (how far above average)
                                    - Cover the time period requested
                                    - Explicitly state if no anomalies were found
                                    Score below 4 if ANY of these are missing.

                                    SEARCH output must:
                                    - Return actual matching transactions with dates and amounts
                                    - Be specific to what was searched for
                                    - Explicitly state if no matches were found
                                    Score below 4 if ANY of these are missing.

                                    THRESHOLD: Score of 4 or above = acceptable. Below 4 = needs revision.
                                    Set needs_revision = True if ANY active agent scores below 4.
                                    Only evaluate agents that were actually invoked (others return None).""")

    user_prompt = HumanMessage( f"""User's original question: {original_question}
                    Active agents: {state.active_agents}
                    Agent outputs: {state.agent_outputs}
                    
                    Evaluate each active agent's output against the rubric.""")

    messages = [system_prompt, user_prompt]

    decision = structured_llm.invoke(messages)

    feedback_dict = {}
    needs_revision = False

    if decision.analyst and decision.analyst.score < 4:
        feedback_dict['analyst'] = {'feedback': decision.analyst.feedback, 'issue': decision.analyst.issue}
        needs_revision = True
    if decision.search and decision.search.score < 4:
        feedback_dict['search'] = {'feedback':decision.search.feedback, 'issue':decision.search.issue}
        needs_revision = True
    if decision.anomaly and decision.anomaly.score < 4:
        feedback_dict['anomaly'] = {'feedback': decision.anomaly.feedback, 'issue': decision.anomaly.issue}
        needs_revision = True

    # For Debugging
    # print(f"[CRITIC DEBUG] analyst score: {decision.analyst.score if decision.analyst else 'N/A'}")
    # print(f"[CRITIC DEBUG] anomaly score: {decision.anomaly.score if decision.anomaly else 'N/A'}")
    # print(f"[CRITIC DEBUG] search score: {decision.search.score if decision.search else 'N/A'}")
    # print(f"[CRITIC DEBUG] needs_revision: {needs_revision}")
    # print(f"[CRITIC DEBUG] revision_count: {state.revision_count}")

    return {
        'needs_revision': needs_revision,
        'revision_feedback': feedback_dict,
        'revision_count': state.revision_count + 1,
    }