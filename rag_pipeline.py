from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_classic.chains import RetrievalQA
import os
import unicodedata
from dotenv import load_dotenv

load_dotenv(override=True)


def _get_provider():
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_accents.lower()


def _is_truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_exhaustive_keywords() -> list[str]:
    raw = os.getenv("RAG_EXHAUSTIVE_KEYWORDS", "liste,tous,complete,capitaines")
    return [keyword.strip().lower() for keyword in raw.split(",") if keyword.strip()]


def _should_force_exhaustive(query: str) -> bool:
    if not _is_truthy_env("RAG_EXHAUSTIVE_MODE", True):
        return False

    normalized_query = _normalize(query)
    keywords = _get_exhaustive_keywords()
    return any(_normalize(keyword) in normalized_query for keyword in keywords)


def _merge_source_documents(first_docs, second_docs):
    merged = []
    seen = set()

    for doc in list(first_docs) + list(second_docs):
        source = doc.metadata.get("source") if doc.metadata else ""
        key = (source, doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    return merged


def _extract_llm_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


class AdaptiveRAGChain:
    def __init__(self, primary_chain, verification_chain, llm):
        self.primary_chain = primary_chain
        self.verification_chain = verification_chain
        self.llm = llm

    def __call__(self, payload):
        query = payload.get("query", "")
        first_pass = self.primary_chain({"query": query})

        if not _should_force_exhaustive(query):
            return first_pass

        verification_query = (
            f"{query}\n\n"
            "Verification pass: produce a complete answer from all relevant entries. "
            "If this is a list request, include every matching item found in context."
        )
        second_pass = self.verification_chain({"query": verification_query})

        merged_docs = _merge_source_documents(
            first_pass.get("source_documents", []),
            second_pass.get("source_documents", []),
        )

        synthesis_prompt = (
            "You are validating a RAG answer with two retrieval passes.\n"
            "Return the final answer in the same language as the user question.\n"
            "If the question requests a list, return an exhaustive list with no missing items from the available context.\n\n"
            f"User question:\n{query}\n\n"
            f"Pass 1 answer:\n{first_pass.get('result', '')}\n\n"
            f"Pass 2 answer:\n{second_pass.get('result', '')}\n\n"
            "Final validated answer:"
        )
        final_answer = _extract_llm_text(self.llm.invoke(synthesis_prompt))

        return {
            "result": final_answer,
            "source_documents": merged_docs,
            "exhaustive_mode": True,
        }


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

    retriever_k = _get_int_env("RAG_RETRIEVER_K", 12)
    retriever_fetch_k = _get_int_env("RAG_RETRIEVER_FETCH_K", 40)
    retriever = vectordb.as_retriever(
        search_type=os.getenv("RAG_RETRIEVER_SEARCH_TYPE", "mmr"),
        search_kwargs={
            "k": retriever_k,
            "fetch_k": retriever_fetch_k,
        },
    )

    exhaustive_k = _get_int_env("RAG_EXHAUSTIVE_RETRIEVER_K", max(20, retriever_k))
    exhaustive_fetch_k = _get_int_env("RAG_EXHAUSTIVE_RETRIEVER_FETCH_K", max(80, retriever_fetch_k))
    verification_retriever = vectordb.as_retriever(
        search_type=os.getenv("RAG_RETRIEVER_SEARCH_TYPE", "mmr"),
        search_kwargs={
            "k": exhaustive_k,
            "fetch_k": exhaustive_fetch_k,
        },
    )

    llm = _build_llm()

    primary_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    verification_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=verification_retriever,
        return_source_documents=True
    )

    return AdaptiveRAGChain(
        primary_chain=primary_chain,
        verification_chain=verification_chain,
        llm=llm,
    )
