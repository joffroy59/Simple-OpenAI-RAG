## TODO

### test azure openai

I’ll quickly inspect how this repo initializes the OpenAI client so I can tell you exactly whether Azure OpenAI can be used as a drop-in replacement and what needs to change.

I’m checking the main pipeline and config files now to confirm what provider-specific assumptions are hardcoded.

Read [](file:///d%3A/dev-data/githome/github/Simple-OpenAI-RAG/rag_pipeline.py)

Read [](file:///d%3A/dev-data/githome/github/Simple-OpenAI-RAG/create_database.py)

Read [](file:///d%3A/dev-data/githome/github/Simple-OpenAI-RAG/README.md)

Read [](file:///d%3A/dev-data/githome/github/Simple-OpenAI-RAG/app.py)

Yes, absolutely.

In this repo, Azure OpenAI is not wired yet, but it can replace the current OpenAI provider with small changes.

Current hardcoded OpenAI usage is in:
- rag_pipeline.py
- create_database.py

What to change:
1. Add an azure provider branch in rag_pipeline.py (alongside openai and ollama) using AzureChatOpenAI.
2. Replace direct OpenAIEmbeddings construction with provider-based embeddings selection (OpenAIEmbeddings vs AzureOpenAIEmbeddings).
3. Do the same in create_database.py so your FAISS index is built with Azure embeddings too.
4. Add Azure env vars in .env:
- LLM_PROVIDER=azure
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_CHAT_DEPLOYMENT
- AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT

Important:
- Azure uses deployment names (not raw model names), so your code should reference deployment fields for chat and embeddings.
- Rebuild the vector DB after switching embedding provider so dimensions stay consistent.

If you want, I can implement these edits directly now in rag_pipeline.py and create_database.py, plus update README.md with Azure setup instructions.