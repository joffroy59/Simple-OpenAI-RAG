from rag_pipeline import create_rag_chain

qa_chain = create_rag_chain()
while True:
    query = input("Ask a question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain({"query": query})
    print("\nAnswer: ", result['result'])