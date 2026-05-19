from pathlib import Path
import sys
curr_dir = Path.cwd()
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))
from Financagent.Phrase_3_MultiAgent.core.tools import get_full_annual_report
from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import datetime as dt
import dotenv
dotenv.load_dotenv()

llm = ChatOpenAI(model = 'gpt-4o')

def report_writer_node(state: FinanceSystemState):

    if not state.needs_report:
        return []

    if state.current_year or state.current_month:

        year = state.current_year if state.current_year else dt.datetime.strptime(state.current_month, "%Y-%m").year

    else:
        return f"Year is not provided. Check the state updating."

    report_data = get_full_annual_report.invoke({'year': year})

    system_prompt = SystemMessage(f"""You are a professional personal finance analyst generating a comprehensive annual financial report.
    Your job is to transform raw transaction data into a clear, insightful, and actionable narrative report.

    REPORT STRUCTURE — always follow this order:
    1. **Executive Summary** 
       - Total annual spending in one sentence
       - 2-3 highest level insights (biggest category, most expensive month, any major anomalies)

    2. **Spending by Category**
       - Rank categories from highest to lowest spending
       - Highlight any category that seems unusually high or low
       - Note any categories that only appear in certain months

    3. **Monthly Trends**
       - Identify the highest and lowest spending months
       - Note any clear seasonal patterns
       - Flag any months with dramatic spikes or drops

    4. **Notable Transactions**
       - Highlight the top 5 largest individual transactions
       - Provide brief context for each (category, when it occurred)

    5. **Anomalies & Unusual Activity**
       - List months where anomalous transactions were detected
       - Briefly explain why they were flagged
       - If no anomalies, state this clearly

    6. **Actionable Observations**
       - Provide exactly 3 concrete, specific observations the user can act on
       - Base these on the actual data patterns — do not give generic financial advice
       - Example: "Your grocery spending peaks in December — consider budgeting $X extra for Q4"

    FORMATTING RULES:
    - Present all amounts as positive numbers with a $ sign
    - Use markdown headers and bullet points for clarity
    - Keep each section concise — insights over raw data repetition
    - Write in second person ("you spent", "your highest category")
    - Professional but accessible tone — avoid jargon

    STRICT RULES:
    - Never fabricate data or insights not supported by the raw data
    - Never give investment advice
    - If data is missing for certain months, acknowledge it rather than ignoring it
    """)

    user_prompt = HumanMessage(f"""Generate a professional annual financial report 
    using this data: {report_data}""")

    messages = [system_prompt, user_prompt]

    response = llm.invoke(messages)

    return {
        'agent_outputs': {'report_writer': response.content},
        'final_report': response.content,
        'messages': [AIMessage(response.content)]
    }

if __name__ == '__main__':
    mock_state = FinanceSystemState(messages=[HumanMessage("Can I get the spending report for 2025?")], needs_report=True, current_year=2025)

    print(report_writer_node(mock_state))