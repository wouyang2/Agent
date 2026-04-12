from dotenv import load_dotenv
from state import State

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import AIMessage

load_dotenv()

# llm = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview")

llm = ChatOpenAI(model = 'gpt-4o-mini')

sys_prompt = """
                You are a professional critic that would be reviewing a report.
                You should determine if the report could properly answer the user question.
                Generate a feedback on what needs to be improved, what is missing, what is good. 
                Make sure the feedback is in clear and readable format.
            """

agent = create_agent(llm, system_prompt=sys_prompt)

def critic(state: State):
    report = state["report"]
    question = state["user_question"]

    user_prompt = f"""Please criticise the report see if it could sufficiently answer the user question. Report: {report}. Question: {question}"""

    result = agent.invoke({
        'messages': [{
            'role':'user',
            'content': user_prompt
        }]
    })

    return {'critique': [AIMessage(content = result['messages'][-1].content)], 'revision_count': 1}