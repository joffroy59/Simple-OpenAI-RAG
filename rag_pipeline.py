from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def create_rag_chain():
    vectordb = FAISS.load_local(
        "faiss_index",
        OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
