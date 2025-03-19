import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_pdfs():
    pdf_chunks = []
    for path in Path("docs").glob("*.pdf"):
        loader = PyPDFLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        pdf_chunks.extend(chunks)
    return pdf_chunks if pdf_chunks else None

def ingest_html():
    html_files = list(Path("docs").glob("*.html"))
    docs = []
    for file in html_files:
        loader = UnstructuredHTMLLoader(str(file))
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    return chunks

def ingest_docs():
    pdf_chunks = ingest_pdfs()
    html_chunks = ingest_html()
    chunks = pdf_chunks + html_chunks
    
    # Initialize Pinecone vector store
    vectorstore = Pinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

    return vectorstore

if __name__ == "__main__":
    ingest_docs()


