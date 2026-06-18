from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def _get_provider():
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _build_embeddings():
    provider = _get_provider()

    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'langchain-ollama'. Install it with 'uv add langchain-ollama' "
                "or 'python -m pip install langchain-ollama'."
            ) from exc

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # Default to OpenAI-compatible embeddings.
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def _build_llm():
    provider = _get_provider()
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'langchain-ollama'. Install it with 'uv add langchain-ollama' "
                "or 'python -m pip install langchain-ollama'."
            ) from exc

        return ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    elif provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=0,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Choose 'openai', 'azure', or 'ollama'.")

def create_rag_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        _build_embeddings(),
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
