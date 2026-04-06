from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

def retriever():

    vectorstore = Chroma(
        persist_directory="./chromaDB",
        collection_name="knowledge_base",
        embedding_function= OpenAIEmbeddings(model = 'text-embedding-3-small')
    )

    # print("check: ",vectorstore._collection.count())

    return vectorstore.as_retriever(search_kwargs = {'k': 3})






