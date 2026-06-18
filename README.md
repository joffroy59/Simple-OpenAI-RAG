# 🧠 Ask Me Anything: OpenAI Research RAG

A **Retrieval-Augmented Generation (RAG)** system built with **LangChain** and **OpenAI**, allowing users to ask natural language questions grounded in OpenAI's latest research papers.

You can easily adjust the knowledge base by modifying the PDFs placed in the `data/` folder — making the context fully customizable and domain-adaptable.

This project demonstrates how to build a modular and powerful RAG pipeline, from document ingestion to semantic retrieval and LLM-based answering. It’s suitable for educational purposes, rapid prototyping, or production-ready extensions.

---

## 📌 Features

- 🔍 Semantic search over embedded PDF content using FAISS
- 🧾 Robust document ingestion & chunking via LangChain + Unstructured
- 🧠 GPT-3.5-powered answers grounded in real documents
- 🧪 Interact through CLI or Streamlit Web UI
- 📚 Preloaded with OpenAI research papers (GPT-4, Whisper, DALL·E 3, etc.)
- ⚡ Optional: swap OpenAI embeddings with HuggingFace to avoid rate limits

---

## 🧑‍💻 Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [Hugging Face Embeddings](https://www.sbert.net/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/openai-rag-system.git
cd openai-rag-system
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Configure your provider (`.env`)

Create a `.env` file in the root directory. This project supports multiple LLM providers via `LLM_PROVIDER`.

#### Option A: OpenAI

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### Option B: Azure OpenAI

Use deployment names from your Azure OpenAI resource:

```
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=your_chat_deployment_name
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=your_embeddings_deployment_name
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

#### Option C: Ollama (local)

```
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

#### PDF Parsing Language (Unstructured)

To avoid the warning `No languages specified, defaulting to English.`, set:

```
UNSTRUCTURED_LANGUAGES=fra
```

You can provide multiple languages if needed:

```
UNSTRUCTURED_LANGUAGES=fra,eng
```

> If you change embedding provider or embedding model, rebuild the FAISS index by running `python create_database.py` again.

> 💡 Don’t forget to add `.env` to your `.gitignore`.

---

## 📥 Add PDF Files

Place research papers in the `data/` folder. For example:

- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- [Whisper](https://cdn.openai.com/papers/whisper.pdf)
- [DALL·E 3](https://cdn.openai.com/papers/dall-e-3.pdf)

---

## ⚙️ Build the Vector Store

Run the following to ingest documents and build the FAISS index:

```bash
python create_database.py
```

> 🛑 **Out of OpenAI quota?** You can switch to local embeddings by editing `create_database.py`:
> ```python
> from langchain_community.embeddings import HuggingFaceEmbeddings
> embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
> ```

---

## 💬 Ask Questions (Two Interfaces)

### ▶️ Option 1: Command Line Interface (CLI)

```bash
python cli.py
```

Example:
```
Ask a question (or type 'exit' to quit): What are the main improvements in GPT-4?
```

---

### 🌐 Option 2: Streamlit Web App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start querying!

---

## 📦 Project Structure

```
├── app.py                # Streamlit interface
├── cli.py                # CLI interface
├── create_database.py    # Ingest and embed documents
├── rag_pipeline.py       # Builds the RAG chain
├── utils.py              # Environment variable loader
├── data/                 # Place PDFs here
├── faiss_index/          # Generated FAISS vector index
├── .env                  # API key (not committed)
└── requirements.txt
```

---

## ✅ To Do / Extensions

- 🔗 Add citations to returned answers
- 📤 Allow file uploads via Streamlit
- ⚡ Add async support or GPU acceleration
- 🐳 Dockerize for easy deployment
- 📈 Add feedback logging and user analytics

---

## 📜 License

MIT License. Free to use, share, and extend.

---

## 🙌 Acknowledgments

- Built with love using [LangChain](https://www.langchain.com/) and [OpenAI](https://openai.com/research).