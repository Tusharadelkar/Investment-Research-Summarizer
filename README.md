# 📈 Investment Research Summarizer

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

A powerful Flask interface for **PDF analytics** and **document-grounded Q&A**. 

This application empowers you to upload financial reports or research papers, automatically extracting text and generating sentence embeddings. It utilizes a hybrid search approach (**BM25 + dense retrieval** with **reciprocal rank fusion**) to pinpoint the most relevant excerpts, which are then passed to a Large Language Model (LLM) to accurately answer your questions.

---

## ✨ Features

- **🔒 User Authentication:** Secure registration and login system.
- **📄 Multi-Format Document Extraction:** Seamlessly parses uploaded PDF, DOCX, and images (with OCR support).
- **⚙️ Background Processing:** Fast and non-blocking document analysis utilizing Redis and RQ workers.
- **🔍 Hybrid Search:** Combines keyword search (BM25) with semantic search (ChromaDB embeddings) for highly accurate context retrieval.
- **🤖 LLM Integration:** Supports multiple providers (Groq, OpenAI, OpenRouter) to answer questions based *only* on the document context.
- **📊 Visual Context:** Displays relevant document pages alongside text answers. Supports Vision LLMs to interpret charts and images directly.

---

## 🚀 Setup & Installation

Follow these steps to run the application locally:

1. **Open a terminal in the project directory.**

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install system dependencies:**
   - [Redis](https://redis.io/) (Required for background workers).
   - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Required for image parsing).

4. **Install Python dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables:**
   ```powershell
   Copy-Item .env.example .env
   ```
   Open the `.env` file and add your API keys:
   - Add a `GROQ_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`.
   - Add `REDIS_URL` (Defaults to `redis://localhost:6379`).
   - *(Optional)* Set `RERANKER_MODEL` to a local model folder for cross-encoder reranking.
   - *(Optional)* Set `VISION_MODEL` or `GROQ_VISION_MODEL` for direct chart and image interpretation.

---

## 💻 Usage

1. **Start the Redis Server:**
   Ensure your local Redis server is running.

2. **Start the Background Worker:**
   Open a new terminal, activate the virtual environment, and run:
   ```powershell
   python worker.py
   ```

3. **Start the Flask server:**
   In your main terminal, run:
   ```powershell
   python app.py
   ```

4. **Access the application:**
   Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

> **Note:** 
> - Uploaded documents are stored in the `uploads/` directory.
> - Document metadata and user accounts are stored locally in `documents.db` (SQLite).
> - Vector embeddings are persisted via ChromaDB in the `chroma_db/` directory.