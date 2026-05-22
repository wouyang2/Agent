import sys
from pathlib import Path
curr_dir = Path(__file__).parent
sys.path.insert(0, str(curr_dir))
from .graph import app
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import uuid
load_dotenv()

def run(user_input: str, thread_id: str):

    config = {'configurable': {'thread_id': thread_id}}

    response = app.invoke(
        {
            'messages':[HumanMessage(user_input)],
            "entities": {},
            "tool_history": [],
            "current_year": None,
            "current_month": None,
            "current_category": None,
            "last_question_type": None,
            "summary": None,
            "agent_outputs": {},
            "routing_decision": [],
            "active_agents": [],
            "needs_report": False,
            "final_response": None,
         },
        config = config
    )

    return {
        'response' : response.get('final_response') or response['messages'][-1].content,
        'routed_to': response.get('active_agents', []),
        'critic_scores': response.get('revision_feedback', {}),
        'revision_count': response.get('revision_count', 0),
        'needs_revision': response.get('needs_revision', False),
        'tool_history': response.get('tool_history', [])[-3:],
    }


def stream_agent(user_input: str, thread_id: str):
    """Generator that yields response text chunks."""
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    for chunk in app.stream(
            {
                "messages": [HumanMessage(user_input)],
                "entities": {},
                "tool_history": [],
                "current_year": None,
                "current_month": None,
                "current_category": None,
                "last_question_type": None,
                "summary": None,
                "agent_outputs": {},
                "routing_decision": [],
                "active_agents": [],
                "needs_report": False,
                "final_response": None,
                "needs_revision": False,
                "revision_count": 0,
                "revision_feedback": {},
            },
            config=config,
            stream_mode="values"
    ):
        final_state = chunk

    # Extract final response
    final_response = final_state.get('final_response') or final_state['messages'][-1].content

    return {
        'response': final_response,
        'routed_to': final_state.get('active_agents', []),
        'revision_count': final_state.get('revision_count', 0),
        'needs_revision': final_state.get('needs_revision', False),
        'tool_history': final_state.get('tool_history', [])[-3:],
    }


# if __name__ == "__main__":
#     thread_id = str(uuid.uuid4())
#     print("Personal Finance Agent — Multi-Agent System")
#     print("Type 'exit' or 'quit' to end\n")
#
#     while True:
#         user_input = input("Ask: ").strip()
#
#         if not user_input:
#             continue
#
#         if user_input.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#
#         response = run(user_input, thread_id)
#         print(f"\nAgent: {response}\n")