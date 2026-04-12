from dotenv import load_dotenv
from state import State

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from datetime import date

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview")

sys_prompt = f"""
                You are professional and detail-oriented writer that would be writing a report that 
                would be answering the user asked question based on the analysis of the raw source and some key quote as well as the corresponding URLs. 
                Reference ALL provided sources in the report, not just one.
                Today's date is {date.today()} for your reference.
                Return the report in a clear, readable format.
                """

agent = create_agent(llm, system_prompt=sys_prompt)

def writer(state : State):
    question = state['user_question']
    analysis = state['analysis']
    sources = state['sources']

    result = agent.invoke({
        'messages' : [{'role': 'user', 'content': f"""Please using the analysis the sources to write a report to answer the user question. Question: {question} Analysis: {analysis}, Sources: {sources}"""}],
    })

    report = result['messages'][-1].content

    return {'report': report}







