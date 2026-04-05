from langchain_openai import OpenAIEmbeddings
import chromadb

from dotenv import load_dotenv
load_dotenv()

def retrieve(query: str, n_result: int = 4):

    client = chromadb.PersistentClient('./chromaDB')
    collections = client.get_or_create_collection(name='knowledge_base')

    embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
    query_embedding = embedding_model.embed_query(query)

    results = collections.query(query_embedding, n_results=n_result)

    return results




