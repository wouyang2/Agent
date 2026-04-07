from dotenv import load_dotenv
from tools.retriever import retriever

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.memory import ConversationSummaryBufferMemory

load_dotenv()

llm = ChatOpenAI()
retriever = retriever()

contextualized_prompt = ChatPromptTemplate.from_messages([
    ('system', """Given the chat history and the lastest user asked question, 
    generate a standalone question that can be understood without the need of history. 
    DO NOT answer the question, just rewrite it. If the question itself is already standalone, then return the question directly"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

qa_prompt = ChatPromptTemplate.from_messages([
    ('system', """You are a knowledge base bot that can answer question using only the context provided.
    IF the relevant context is not provided, just say I do not know.
    
    Context: {context}"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_prompt)
combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,combine_docs_chain)

# chat_history = []

chat_history = ConversationSummaryBufferMemory(llm = llm, max_token_limit=500, return_messages=True)

# while True:
#     query = input('Enter query: ')
#
#     if query.lower() in ['exit', 'quit']:
#         break
#
#     response = rag_chain.invoke({'input': query , 'chat_history': chat_history})
#
#     print('AI response: ', response['answer'])
#
#     chat_history.extend([HumanMessage(content=query), AIMessage(content= response['answer'])])

while True:
    query = input('Enter query: ')

    if query.lower() in ['exit', 'quit']:
        break

    response = rag_chain.invoke({'input': query , 'chat_history': chat_history.load_memory_variables({})['history']})

    print('AI response: ', response['answer'])

    chat_history.save_context({'input' : query}, {'output' : response['answer']})


