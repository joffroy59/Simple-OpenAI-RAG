# 🧠 Ask Me Anything: OpenAI Research RAG

A **Retrieval-Augmented Generation (RAG)** system built with **LangChain** and **OpenAI**, allowing users to ask natural language questions grounded in OpenAI's latest research papers.
Do note that you can adjust the RAG context by changing the PDFs fed into the `data` folder

This project demonstrates how to build a simple, modular, and powerful RAG pipeline, from document ingestion to retrieval to LLM-based answering, and is suitable for publishing or extending in production systems.

---

## 📌 Features

- 🔍 Semantic document search using vector embeddings (FAISS)
- 🧾 PDF ingestion & chunking using LangChain
- 🧠 GPT-3.5 powered question answering with document-grounded answers
- 🧪 Query via CLI or a Streamlit web UI
- 📚 Preloaded with OpenAI research papers (GPT-4, Whisper, DALL·E 3, etc.)

---

## 🧑‍💻 Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Python-dotenv](https://pypi.org/project/python-dotenv/)
- [Unstructured](https://github.com/Unstructured-IO/unstructured) for PDF parsing

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
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

> 💡 Don’t forget to add `.env` to your `.gitignore`.

---

## 📥 Add PDF Files

Place your research papers in the `data/` directory. Some suggested PDFs:
- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- [Whisper](https://cdn.openai.com/papers/whisper.pdf)
- [DALL·E 3](https://cdn.openai.com/papers/dall-e-3.pdf)

---

## ⚙️ Build the Vector Store

Run the following to ingest and embed documents:

```bash
python ingest_documents.py
```

---

## 💬 Ask Questions (Two Options)

### Option 1: CLI

```bash
python cli.py
```

Example:
```
Ask a question (or type 'exit' to quit): What are the main improvements in GPT-4?
```

### Option 2: Streamlit Web App

```bash
streamlit run app.py
```

Access the app in your browser and type in any question!

---

## 📦 Project Structure

```
├── app.py                # Streamlit UI
├── cli.py                # Command-line interface
├── rag_pipeline.py       # Builds the RAG chain
├── ingest_documents.py   # Loads & embeds PDF documents
├── utils.py              # .env loader
├── data/                 # Store PDFs here
├── faiss_index/          # FAISS vector store (auto-generated)
├── .env
└── requirements.txt
```

---

## ✅ To Do / Extensions

- Add citations to source documents
- Support user-uploaded PDFs
- Switch to async embedding or GPU
- Deploy via Docker / Streamlit Cloud

---

## 📜 License

MIT License. Feel free to use and modify this project.

---

## 🙌 Acknowledgments

- Built with love using [LangChain](https://www.langchain.com/) and [OpenAI](https://openai.com/research).