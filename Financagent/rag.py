from langchain_openai import  OpenAIEmbeddings
from langchain_chroma import Chroma

import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

embd = OpenAIEmbeddings(model = "text-embedding-3-small")

def format_document(row):
    direction = "debit" if row.IS_DEBIT else "credit card payment"
    amount = abs(row.AMOUNT)
    return f"${amount:.2f} {direction} at {row.DESCRIPTION} on {row.DATE}, category: {row.CATEGORY}"

# For filter queries
def build_metadata(row) -> dict:
    # idx, date, desc, amount, status, is_debit, month, year, category = row

    metadata = {
        'amount' : float(row.AMOUNT),
        'date' : str(row.DATE),
        'month' : str(row.MONTH),
        'year' : str(row.YEAR),
        'category' : str(row.CATEGORY).lower(),
        'is_debit' : str(row.IS_DEBIT).lower(),
    }

    return metadata

def index_transaction(file_path, persist_dir):

    # Load the normalized file
    try:
        with open(file_path, 'r') as file:
            df = pd.read_csv(file)

    except FileNotFoundError:
        raise FileNotFoundError("File not found")

    # Initialize a vector store
    vct_store = Chroma(collection_name="transactions", embedding_function=embd, persist_directory=persist_dir)

    documents = []
    md = []

    for row in df.itertuples():

        # build documents and metadata
        document = format_document(row)
        documents.append(document)
        md.append(build_metadata(row))

    ids = ['tsc_' + str(i) for i in range(len(df))]

    assert len(documents) == len(md) == len(ids)

    if os.path.exists(persist_dir):
        vct_store.reset_collection()

    vct_store.add_texts(texts=documents, metadatas=md, ids=ids)

    print("Take a look on the first 3 data in the collection: ", vct_store._collection.peek(limit=3))

# FOR AGENT CALLING
def query_transactions(question: str, collection, n_results: int = 10, filters:dict = None):

    kwargs = {'query_texts' : question, 'n_results' : n_results}
    if filters:
        kwargs['where'] = filters

    # result = collection.query(**kwargs)
    result = collection.similarity_search(
        query=question,
        k= n_results,
        filter=filters,
    )
    return result

if __name__ == "__main__":

    curr_dir = os.path.dirname(__file__)
    file_path = os.path.join(curr_dir, "data/processed/categorized_data.csv")

    persist_path = os.path.join(curr_dir, "chroma_db")

    index_transaction(file_path, persist_path)


