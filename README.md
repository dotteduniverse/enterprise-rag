Project Title: Enterprise Agentic RAG API

# Enterprise Agentic RAG API

A production-ready, self-correcting AI Agent built with Python, FastAPI, and Pydantic. This project demonstrates the **LOTMG** architecture (LLM, Orchestrator, Tools, Memory, Guardrails) designed for enterprise banking and compliance environments.

## Architecture Diagram

```text
[ User Prompt ]
      │
      ▼
[ Orchestrator (While Loop) ] ──(Checks STOW Budgets: Steps, Tokens, Cost, Time)
      │
      ├─► [ LLM (Groq/Llama-3) ] ──(Outputs JSON)──┐
      │                                            │
      ▼                                            ▼
[ Parse JSON ] ──► [ Tool Registry ] ──► [ Pydantic Validation ]
                                               │
                                     (Pass) ───┴─── (Fail)
                                       │              │
                                       ▼              ▼
                                  [ Policy Gate ] [ Return Error 
                                       │           as Observation ]
                                       ▼              │
                                  [ Execute ]         │
                                     Tool ]           │
                                       │              │
                                       └──────┬───────┘
                                              ▼
                                    [ Append to Memory ]
                                              │
                                              ▼
                                         [ Next Loop ]

```

Features
Hybrid Search: Combines Dense (Vector) and Sparse (BM25) retrieval with a Cross-Encoder re-ranker for high precision.

Agentic Orchestrator: Uses a while loop with STOW budgets (Steps, Tokens, Cost, Wall-clock) to prevent runaway loops.

Error-as-Observation: If the LLM hallucinates a parameter, Pydantic rejects it. The Orchestrator feeds the exact error back to the LLM, forcing self-correction.

Pre-Action Policy Gate: Intercepts unauthorized tool calls (e.g., Confused Deputy attacks) before they hit the database.

Conversation Memory: Maintains context across multi-turn conversations.

Tech Stack
Python & FastAPI (API Layer)

ChromaDB (Vector Database)

SentenceTransformers (Local Embeddings & Cross-Encoder Re-ranker)

Pydantic (Strict Schema Validation)

Groq / Cerebras (LLM Providers via OpenAI SDK)

How to Run
Clone the repo and navigate to the folder.

Create a virtual environment and install dependencies:

bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Add your API keys to a .env file:

env
GROQ_API_KEY=your_key_here
CEREBRAS_API_KEY=your_key_here
Ingest a PDF: python ingestion.py

Start the API: python api.py

Test via Swagger UI: http://localhost:8000/docs
.
