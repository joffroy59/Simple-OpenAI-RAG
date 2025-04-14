from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

DATA_FOLDER = "data"
def load_and_split(pdf_path):
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split(documents)
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