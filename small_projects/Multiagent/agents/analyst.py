from state import State
from dotenv import load_dotenv
from typing_extensions import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, field_validator

load_dotenv()

class Analyze_Output_Format(BaseModel):
    analysis : str
    sources : dict[str , list[str]]

    @field_validator('sources', mode='before')
    @classmethod
    def validate_sources(cls, v):

        return {
            key: [val] if isinstance(val, str) else val
            for key, val in v.items()
        }

# llm = ChatAnthropic(model = 'claude-sonnet-4-5')

llm = ChatOpenAI(model = 'gpt-4o-mini')

sys_prompt = """
    You are a professional analyst reviewing raw research sources.

    Your job:
    1. Carefully analyze the provided sources and the original research question
    2. Summarize the key findings clearly and concisely
    3. Extract key quotes from each source

    When returning sources, format them as a dict where:
    - Key is the FULL URL of the source
    - Value is a LIST of key quote strings extracted from that source

    Example of correct format:
    {
        "https://aws.amazon.com/what-is-llm": [
            "LLMs are trained on massive datasets",
            "They use transformer architecture to process text"
        ],
        "https://ibm.com/large-language-models": [
            "LLMs can generate human-like responses",
            "Fine-tuning adapts LLMs to specific tasks"
        ]
    }

    Rules:
    - Always use the full URL as the key, never the source name
    - Always wrap quotes in a list, even if there is only one quote
    - Only include quotes that are directly relevant to the research question
    - If a source has no relevant quotes, exclude it entirely
"""

agent = create_agent(llm, system_prompt= sys_prompt, response_format=Analyze_Output_Format)

def analyze_sources(state: State):
    question = state['user_question']
    raw_text = state['search_result']

    result = agent.invoke({
        'messages': [{
            'role': 'user',
            'content': f"""Given the raw text and the user asked question, please analyze the raw text. Raw Text: {raw_text}, Question: {question}"""
        }]
    })

    response = result['structured_response']
    analysis = response.analysis
    sources = response.sources

    return {'analysis': analysis, 'sources': sources}




