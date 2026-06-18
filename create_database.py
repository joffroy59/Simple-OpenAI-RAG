from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv(override=True)
DATA_FOLDER = "data"


def _get_unstructured_languages():
    raw_languages = os.getenv("UNSTRUCTURED_LANGUAGES", "fra")
    return [lang.strip() for lang in raw_languages.split(",") if lang.strip()]


def _build_embeddings():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

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

    # OpenAI-compatible default embeddings.
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def load_and_split(pdf_path):
    loader = UnstructuredPDFLoader(
        pdf_path,
        languages=_get_unstructured_languages(),
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks

def build_vectorstore(chunks):
    embeddings = _build_embeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local("faiss_index")

if __name__ == "__main__":
    all_chunks = []
    for file_name in os.listdir(DATA_FOLDER):
        if file_name.endswith(".pdf"):
            print(f"Processing {file_name}")
            pdf_path = os.path.join(DATA_FOLDER, file_name)
            chunks = load_and_split(pdf_path)
            all_chunks.extend(chunks)
    build_vectorstore(all_chunks)
    print("Database created successfully")