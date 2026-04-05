from dotenv import load_dotenv
from tools.retriever import retrieve
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
import aisuite

chat_history = ChatMessageHistory()
CLIENT = aisuite.Client()

while True:
    query = input('Enter query: ')

    if query.lower() in ['exit', 'quit']:
        break

    # history.add_user_message(query)

    raged_query = retrieve(query)  # Retrieve relevant chunk for ChromaBD
    context = '\n\n'.join(raged_query)

    response = CLIENT.chat.completions.create(
        model='openai:gpt-4o-mini',
        messages=[{'role': 'system', 'content': 'Answer using only the context provided.'},
                  {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}],
    )

    print(response.choices[0].message.content)

    # history.add_ai_message(response.choices[0].message)