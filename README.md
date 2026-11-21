# RAG System — PDF Transaction Analyzer

A Retrieval-Augmented Generation (RAG) system that ingests PDF bank/transaction statements, extracts credit and debit entries, stores them as vector embeddings, and answers natural-language queries about transactions via a chat interface.

---

## Project Structure

```
RAG_System/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py          # Extract text/tables from PDFs via pdfplumber
│   │   ├── preprocessor.py        # Clean & normalize transaction text
│   │   └── chunker.py             # Split documents into overlapping chunks
│   ├── embedding/
│   │   ├── embedder.py            # Generate vector embeddings (OpenAI)
│   │   └── vector_store.py        # Store/query embeddings (ChromaDB)
│   ├── extraction/
│   │   ├── transaction_parser.py  # Parse credit/debit entries via regex
│   │   ├── schema.py              # Pydantic models for transactions
│   │   └── classifier.py          # Filter/summarize by type, status, date
│   ├── retrieval/
│   │   ├── retriever.py           # Semantic search over vector chunks
│   │   └── reranker.py            # Keyword-boost reranker
│   ├── generation/
│   │   ├── prompt_builder.py      # Build prompts with retrieved context
│   │   └── llm_client.py          # LLM API calls (OpenAI gpt-4o-mini)
│   ├── api/
│   │   ├── main.py                # FastAPI app entry point + static UI
│   │   ├── routes.py              # REST endpoints
│   │   └── models.py              # Request/response schemas
│   └── utils/
│       ├── config.py              # Config & env vars (pydantic-settings)
│       └── logger.py              # Loguru logging setup
├── static/
│   └── index.html                 # Chat UI (served at http://localhost:8025/)
├── data/
│   ├── raw/                       # Uploaded PDF files
│   ├── processed/                 # Extracted transactions as JSON (per document)
│   └── vector_db/                 # Persisted ChromaDB vector store
├── tests/
│   ├── test_pdf_loader.py
│   ├── test_transaction_parser.py
│   └── test_retrieval.py
├── requirements.txt
├── .env
└── README.md
```

---

## Data Flow

### Upload / Ingestion (`POST /ingest`)

```
PDF Upload
    │
    ▼
[pdf_loader.py]              ← pdfplumber: extract text, tables, metadata
    │
    ▼
[preprocessor.py]            ← Normalize dates, amounts, currency; clean noise
    │
    ▼
[chunker.py]                 ← Split into overlapping chunks (800 chars / 150 overlap)
    │
    ├──► [embedder.py]       ← text-embedding-3-small (OpenAI)
    │         │
    │         ▼
    │    [vector_store.py]   ← ChromaDB: persist embeddings to data/vector_db/
    │
    └──► [transaction_parser.py]  ← Regex extraction of credit/debit rows
              │
              ▼
         [data/processed/<doc_id>.json]  ← Structured transaction list
```

### Query / Chat (`POST /query`)

```
User Question
    │
    ▼
[retriever.py]       ← Embed query → cosine search in ChromaDB → top-K chunks
    │
    ▼
[reranker.py]        ← Keyword-boost rerank, keep top-5
    │
    ▼
[prompt_builder.py]  ← Inject chunks as context; enforce strict no-hallucination rules
    │
    ▼
[llm_client.py]      ← gpt-4o-mini generates structured answer from context
    │
    ▼
API Response         ← { answer, transactions[], sources[] }
```

> **Note:** The `/query` endpoint searches the **vector DB** (ChromaDB) — not the processed JSON.
> The processed JSON in `data/processed/` is only used by `GET /transactions`.

---

## Tech Stack

| Layer        | Tool                                  |
|--------------|---------------------------------------|
| PDF parsing  | `pdfplumber`                          |
| Embeddings   | `text-embedding-3-small` (OpenAI)     |
| Vector DB    | `ChromaDB` (local persistence)        |
| LLM          | `gpt-4o-mini` (OpenAI)               |
| API          | `FastAPI` + `uvicorn`                 |
| Validation   | `Pydantic v2`                         |
| Testing      | `pytest`                              |

---

## Setup

### 1. Prerequisites

- Python 3.13
- OpenAI API key (for embeddings + LLM)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env` and fill in your API keys:

```env
OPENAI_API_KEY=sk-...          # Required for embeddings and LLM
ANTHROPIC_API_KEY=sk-ant-...   # Optional (not used by default)

LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

CHROMA_PERSIST_DIR=data/vector_db
CHROMA_COLLECTION_NAME=transactions

SIMILARITY_THRESHOLD=0.3
TOP_K_RESULTS=10

API_HOST=0.0.0.0
API_PORT=8025
```

### 4. Run the server

```bash
python -m uvicorn src.api.main:app --reload --port 8025
```

Or via the module entry point:

```bash
python -m src.api.main
```

### 5. Open the Chat UI

Navigate to [http://localhost:8025](http://localhost:8025) in your browser.

---

## Usage

### Via Chat UI

1. Open [http://localhost:8025](http://localhost:8025)
2. Click **Upload PDF** and select your bank statement
3. Wait for ingestion to complete (shown in the chat)
4. Type questions in the chat box, e.g.:
   - "Show all debit transactions"
   - "What is the total credit amount?"
   - "List transactions from January 2025"

### Via API (curl)

**Ingest a PDF:**
```bash
curl -X POST http://localhost:8025/ingest \
  -F "file=@statement.pdf"
```

**Query transactions:**
```bash
curl -X POST http://localhost:8025/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show all debits above 5000"}'
```

**List all extracted transactions:**
```bash
curl http://localhost:8025/transactions
```

**Health check:**
```bash
curl http://localhost:8025/health
```

---

## API Endpoints

| Method | Endpoint                  | Description                              |
|--------|---------------------------|------------------------------------------|
| GET    | `/`                       | Chat UI (served from static/index.html)  |
| POST   | `/ingest`                 | Upload and process a PDF                 |
| POST   | `/query`                  | Ask a natural language question          |
| GET    | `/transactions`           | List all extracted transactions          |
| GET    | `/transactions/{id}`      | Get a specific transaction by ID         |
| GET    | `/health`                 | Health check                             |

Interactive API docs: [http://localhost:8025/docs](http://localhost:8025/docs)

---

## Example Query & Response

**Request:**
```json
POST /query
{ "query": "What are my total debits?" }
```

**Response:**
```json
{
  "answer": "Based on the provided context, the total debit amount is 12,450.00.",
  "transactions": [
    {
      "id": "txn_001",
      "transaction_date": "2025-01-15",
      "description": "Amazon Purchase",
      "amount": 1200.00,
      "transaction_type": "debit",
      "status": "completed",
      "balance": 48800.00,
      "currency": "INR"
    }
  ],
  "sources": ["statement_jan2025.pdf p.2"]
}
```

---

## Transaction Schema

All transactions use these exact field names (both in JSON output and API responses):

| Field              | Type              | Description                                     |
|--------------------|-------------------|-------------------------------------------------|
| `id`               | string            | Unique transaction identifier                   |
| `transaction_date` | string (YYYY-MM-DD) | Date of transaction                           |
| `description`      | string            | Transaction narrative/merchant name             |
| `amount`           | number            | Amount (no currency symbols, 2 decimal places)  |
| `transaction_type` | string            | `"credit"` or `"debit"` or `"unknown"`          |
| `status`           | string            | `"completed"`, `"pending"`, `"failed"`, `"reversed"`, `"unknown"` |
| `balance`          | number or null    | Closing balance after transaction               |
| `currency`         | string            | e.g. `"INR"`, `"USD"`                          |

---

## Data Storage

| Store                   | Location              | Purpose                          |
|-------------------------|-----------------------|----------------------------------|
| ChromaDB vector store   | `data/vector_db/`     | Semantic search for `/query`     |
| Extracted transactions  | `data/processed/`     | Structured listing for `/transactions` |
| Raw PDF files           | `data/raw/`           | Original uploaded PDFs           |

---

## LLM Rules (Hallucination Prevention)

The system prompt enforces strict rules on the LLM:

1. Answer **only** using information explicitly present in the retrieved context
2. Never invent, guess, or assume transactions
3. Return `"The provided context does not contain this information."` when data is absent
4. Omit any transaction with missing date, amount, or description
5. Use exact field names as specified in the schema above
6. Format amounts as numbers with 2 decimal places (no commas, no symbols)
