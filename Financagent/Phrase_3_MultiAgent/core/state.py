import dataclasses
from typing import Annotated, List, Optional
from langgraph.graph.message import  add_messages

def merge_dict(a: dict, b: dict) -> dict:
    return a|b

def extend_list(a: list, b: list) -> list:
    return a+b[-10:]

@dataclasses.dataclass
class FinanceSystemState:
    messages: Annotated[List, add_messages]
    current_year: Optional[int] = None
    current_month: Optional[str] = None
    current_category: Optional[str] = None
    last_question_type: Optional[str] = None
    summary: Optional[str] = None
    entities: dict = dataclasses.field(default_factory=dict)
    tool_history: Annotated[list, extend_list] = dataclasses.field(default_factory=list)

    routing_decision: list = dataclasses.field(default_factory=list)  # 'Analyst' 'anomaly' 'search' 'all' 'report'
    active_agents: list = dataclasses.field(default_factory=list)
    agent_outputs: Annotated[dict, merge_dict] = None   # key as agent and the value as input
    needs_report: bool = dataclasses.field(default=False)
    final_response: Optional[str] = dataclasses.field(default=None)

    revision_count : int = 0
    needs_revision: bool = dataclasses.field(default=False)
    revision_feedback : Annotated[dict, merge_dict]= dataclasses.field(default_factory=dict)    # key to be the agent, value would be the str of revision suggestion


