import dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.tools import PythonREPLTool
from langgraph.checkpoint.memory import MemorySaver

from tools.search import tavily_tool
from tools.wiki import wiki_tool


dotenv.load_dotenv()

llm = ChatOpenAI(model = 'gpt-4o-mini')
tools = [tavily_tool(), wiki_tool(), PythonREPLTool()]

system_prompt = """You are a research assistant and you can use provided tools to do the research for me. 
Before taking any action, always:
1. Break down the question into sub-questions
2. Plan which tools you need and in what order
3. Execute your plan step by step
4. Reflect on whether your answer is complete before responding

Please summarize everything as result.
"""

memory = MemorySaver()

agent = create_agent(llm,
                     tools,
                     system_prompt= system_prompt,
                     debug=True,
                     checkpointer= memory,
                     )

config = {"configurable": {'thread_id': '1'}}


while True:

    query = input('> ')

    if query == 'exit':
        break

    response = agent.invoke(
        {'messages': [HumanMessage(query)]},
        config = config,
    )


    print(response['messages'][-1].content)

