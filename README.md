# AI Document Assistant

A PDF-based intelligent document assistant built with FastAPI, LangChain, Cohere, and Groq. Upload any PDF file and securely query its contents — the model answers strictly using only the data found in the document.

---

## Features

- Upload any PDF document through the web interface
- Answers are strictly grounded in the uploaded document context
- Refuses to answer if the query is not covered in the PDF
- Persistent FAISS vector store — index survives server restarts
- Clean modular FastAPI architecture

---

## Project Structure

```
healthbot/
├── app/
│   ├── main.py                 # FastAPI application entrypoint and lifespan
│   ├── api/
│   │   └── routes.py           # /upload and /chat API endpoints
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── pdf_service.py      # PDF text extraction and chunking
│       └── rag_service.py      # FAISS, Cohere embeddings, Groq LLM, strict prompting
├── static/
│   └── index.html              # Frontend UI (PDF upload + chat)
├── data/
│   └── faiss_index/            # Auto-generated FAISS vector store (after first upload)
├── .env                        # API keys (not committed to git)
├── .gitignore
└── requirements.txt
```

---

## How It Works

### PDF Upload Flow

1. User selects a PDF and clicks Upload PDF in the UI
2. `POST /api/upload` receives the file
3. `pdf_service.py` extracts text from each page using PyPDF2
4. Text is split into overlapping 1000-character chunks
5. `rag_service.py` sends chunks to the Cohere API to generate vector embeddings
6. Embeddings are stored and persisted to `data/faiss_index/` using FAISS

### Chat Flow

1. User types a question and clicks Send
2. `POST /api/chat` receives the question
3. FAISS retrieves the top 4 most semantically relevant document chunks
4. Chunks are injected as context into a strict prompt
5. Groq (Llama 3.1) generates an answer using only the provided context
6. If the answer is not in the document, the model refuses to respond

---

## Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Backend          | FastAPI + Uvicorn                 |
| LLM              | Groq (llama-3.1-8b-instant)       |
| Embeddings       | Cohere (embed-english-light-v3.0) |
| Vector Store     | FAISS (local, persisted to disk)  |
| PDF Parsing      | PyPDF2                            |
| Text Splitting   | LangChain RecursiveCharacterTextSplitter |
| Frontend         | Vanilla HTML/CSS/JS               |

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/dineshbalan-b/healthbot.git
cd healthbot
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

- Cohere API key: https://dashboard.cohere.com/api-keys
- Groq API key: https://console.groq.com/keys

Both services offer free tiers.

### 5. Run the Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at: http://localhost:8000

---

## Usage

1. Open http://localhost:8000 in your browser
2. Click "Choose File" and select a medical PDF
3. Click "Upload PDF" and wait for the confirmation message
4. Type your question in the chat input and press Send
5. The assistant will answer based strictly on the uploaded document

---

## Notes

- The `.env` file is excluded from git via `.gitignore` — never commit your API keys
- The `data/faiss_index/` directory is auto-created on first upload
- Uploading a new PDF adds to the existing knowledge base (merges with FAISS index)
- To reset the knowledge base, delete the `data/` folder and restart the server
