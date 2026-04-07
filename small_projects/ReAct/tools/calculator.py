from langchain_classic.chains import LLMMathChain
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import Tool
from langchain_openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI()

def calculator_tool():
    problem_chain = LLMMathChain(llm = llm)
    calculator = Tool.from_function(
        name = "Calculator",
        func = problem_chain.run,
        description = "Useful when you need to solve a math problem",
    )

    return calculator