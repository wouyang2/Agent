from typing_extensions import TypedDict, Annotated, List, Dict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage, SystemMessage
from operator import add

def merge_sources(a: dict, b: dict) -> dict:
    for url, quotes in b.items():
        if url in a:
            a[url].extend(quotes)  # merge quotes for same URL
        else:
            a[url] = quotes
    return a


class State(TypedDict):

    user_question: str
    search_result: Annotated[List, add]

    analysis: str
    sources: Annotated[dict, merge_sources] # Keep the sources URL and key quotes maybe using dict{ URL: list[quotes]}?

    report: str

    critique : Annotated[List, add_messages] # Contains the lastest two messages
    critique_summary: str  # Store the condensed messages so far

    revision_count: Annotated[int, add]


def summarizing_message (state: State):
    llm = ChatOpenAI(model='gpt-4o-mini')
    existing_summary = state['critique_summary']
    messages = state['critique']

    if len(messages) > 10:

        messages_to_summarize = messages[:-2]

        prompt = f"""Summarize the following messages using the provided information. Existing summary: {existing_summary}, New messages to be summarized: {messages_to_summarize}"""

        response = llm.invoke(prompt)

        delete_messages = [RemoveMessage(m.id) for m in messages_to_summarize]

        return {
            "critique_summary": response.content,
            "critique": delete_messages
        }

    return {"critique_summary": existing_summary}



