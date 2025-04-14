import streamlit as st
from rag_pipeline import create_rag_chain

st.title("Ask me anything about OpenAI Research")
query = st.text_input("Ask a question...")

if query:
    qa_chain = create_rag_chain()
    response = qa_chain({"query": query})
    st.write("**Answer:**", response['result'])