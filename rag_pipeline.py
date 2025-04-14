from langchain.chains iomport RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_rag_chain():
    vectordb = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_kwargs{"k" : 4})

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain