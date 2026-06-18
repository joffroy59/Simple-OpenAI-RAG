from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    if provider == "ollama":
        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Choose 'openai' or 'ollama'.")

def create_rag_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = _build_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
