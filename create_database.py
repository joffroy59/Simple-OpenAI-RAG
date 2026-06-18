from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
DATA_FOLDER = "data"

def load_and_split(pdf_path):
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
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