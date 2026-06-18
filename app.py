import streamlit as st
from rag_pipeline import create_rag_chain
from collections import Counter
import os


def _extract_source_files(source_documents):
    files = []
    for doc in source_documents:
        source_path = doc.metadata.get("source") if doc.metadata else None
        if source_path:
            files.append(source_path)
    return files

st.title("Ask me anything about OpenAI Research")
query = st.text_input("Ask a question...")

if query:
    qa_chain = create_rag_chain()
    response = qa_chain({"query": query})
    st.write("**Answer:**", response["result"])

    source_documents = response.get("source_documents", [])
    source_files = _extract_source_files(source_documents)

    if source_files:
        file_counts = Counter(source_files)
        top_source_file, top_count = file_counts.most_common(1)[0]

        st.write("**Main source file used by RAG:**", os.path.basename(top_source_file))
        st.caption(f"Selected in {top_count} retrieved chunk(s)")

        with st.expander("Show retrieved source files"):
            for file_path, count in file_counts.most_common():
                st.write(f"- {os.path.basename(file_path)} ({count} chunk(s))")
    else:
        st.caption("No source file metadata found in retrieved documents.")