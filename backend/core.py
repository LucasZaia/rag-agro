import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Any
from langchain.chains import create_history_aware_retriever
load_dotenv()

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    agro_search = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    rephrased_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=agro_search.as_retriever(), prompt=rephrased_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )

    response = qa.invoke({"input": query, "chat_history": chat_history})
    new_response = {
        "query": response["input"],
        "result": response["answer"],
        "source_documents": response["context"]
    }
    return new_response


if __name__ == "__main__":
    res = run_llm(query="Como identificar doençs na soja?")
    print(res["result"])