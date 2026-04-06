from pathlib import Path
import dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, WebBaseLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

dotenv.load_dotenv()

# File Loader
def load_files():
    base_path = Path(__file__).resolve().parent
    document_path = base_path / 'documents'

    pdf_loader = DirectoryLoader(str(document_path), glob='*.pdf', show_progress=True, loader_cls=UnstructuredPDFLoader)
    txt_loader = DirectoryLoader(str(document_path), glob='*.txt', show_progress=True, loader_cls=TextLoader)
    markdown_loader = DirectoryLoader(str(document_path), glob= '*.md', show_progress=True, loader_cls=UnstructuredMarkdownLoader)

    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://docs.python.org/3/tutorial/datastructures.html",
        "https://python.langchain.com/docs/concepts/vectorstores/",
        "https://platform.openai.com/docs/guides/embeddings"
    ]

    web_loader = WebBaseLoader(urls)

    docs = pdf_loader.load() + txt_loader.load() + markdown_loader.load() + web_loader.load()

    return docs

# Chunking content
def chunk_content(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    return chunks

def embedding_and_storing(chunks):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory='./chromaDB', collection_name='knowledge_base')


def workflow():
    docs = load_files()
    chunks = chunk_content(docs)
    embedding_and_storing(chunks)

if __name__ == "__main__":
    workflow()





