from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from state import State
from dotenv import load_dotenv
load_dotenv()


# llm = ChatAnthropic(model='claude-haiku-4-5-20251001')
llm = ChatOpenAI(model = 'gpt-4o-mini')

tavily_search = TavilySearch()

sys_prompt = f"""
                    You are a professional and detail-oriented research assistant.
                    Do research based on given topic, find maximum 5 resources and return raw research text with the url. 
                """

agent = create_agent(
        model = llm,
        system_prompt=sys_prompt,
        tools=[tavily_search],
    )

def search_agent(state: State):

    question = state['user_question']

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    return {'search_result': [result['messages'][-1].content]}
