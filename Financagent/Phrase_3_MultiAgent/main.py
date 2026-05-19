from graph import app
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

    return response.get('final_response') or response['messages'][-1].content


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    print("Personal Finance Agent — Multi-Agent System")
    print("Type 'exit' or 'quit' to end\n")

    while True:
        user_input = input("Ask: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = run(user_input, thread_id)
        print(f"\nAgent: {response}\n")