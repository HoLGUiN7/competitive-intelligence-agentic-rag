# Competitive Intelligence Agentic RAG (`n8n` + `Qdrant` + `Ollama` + `Docling` + `Crawl4AI`)

A **7-layer Multimodal Agentic RAG system** for Competitive Intelligence and Market Analysis built on a **100% free / local-first stack** (Docker local services + free-tier LLM inference) and quantitatively audited with **Ragas**.

## Architecture Overview (7 Layers)

1. **Data Extraction:** **Crawl4AI** (dynamic web scraping into *Fit Markdown*) + **Docling** (layout-aware PDF parsing preserving complex tables and multi-column layouts).
2. **Text Embeddings:** **Ollama** running `nomic-embed-text` locally (768 dimensions, Cosine similarity).
3. **Vector Database:** **Qdrant** (`inteligencia_competitiva` collection with payload metadata filtering by `competidor` and `categoria`).
4. **Orchestration & Frameworks:** **n8n** workflows leveraging LangChain nodes (`AI Agent`, `Text Classifier`, `Recursive Character Text Splitter`, `Qdrant Vector Store`).
5. **Open LLMs & Inference Access:** **Groq** (low-latency routing and fast factual lookups) + **Ollama** (local embeddings).
6. **LLMs & Model Routing:** Dynamic routing via **Text Classifier** into:
   - **`SIMPLE` route:** Fast factual agent powered by **Groq** (`openai/gpt-oss-120b`).
   - **`COMPLEX` route:** Analytical **Self-RAG** agent powered by **Gemini 2.5 Flash** with iterative context evaluation and Markdown comparison tables.
7. **Evaluation:** **Ragas** evaluation suite in Python (`evaluation/evaluate_rag.py`) measuring context grounding and anti-hallucination behavior.

## Repository Structure

```text
competitive-intelligence-agentic-rag/
├── docker-compose.yml              # Local stack: n8n, Qdrant, Ollama, Crawl4AI, Docling
├── .env.example                    # Environment variable template (no secrets)
├── .gitignore                      # Ignores .env, .venv, caches, and OS files
├── README.md
├── workflows/
│   ├── 01 - Multimodal RAG Ingestion (Crawl4AI + Docling).json
│   └── 02 - Agentic RAG with Model Routing (Groq + Gemini).json
├── pdfs/                           # Benchmark PDFs processed by Docling
└── evaluation/
    ├── eval_dataset.json
    ├── evaluate_rag.py
    ├── ragas_results.json
    └── requirements.txt