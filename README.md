# 🏥 Agentic Research Copilot for Clinical Documents

An AI-powered **RAG (Retrieval-Augmented Generation) agent** that answers complex, multi-hop questions over a corpus of clinical documents — with built-in **answer verification**, **contradiction detection**, and the ability to **abstain** when evidence is too weak.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)
![Claude](https://img.shields.io/badge/Claude-3_Haiku-purple)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-orange)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌─────────────┐
│ Retrieve │────▶│  Verify  │────▶│ Contradict  │
│  (FAISS) │     │(LLM Judge)│    │ (Pairwise)  │
└─────────┘     └──────────┘     └─────────────┘
    ▲                │                    │
    │          needs more?          confidence?
    │                │                    │
    │    ┌───────────┘              ┌─────┴─────┐
    │    ▼                          ▼           ▼
    │ ┌──────────────┐        ┌─────────┐ ┌─────────┐
    └─┤ Reformulate  │        │ Abstain │ │ Answer  │
      │ (Multi-Hop)  │        │   ⚠️    │ │  ✅📚   │
      └──────────────┘        └─────────┘ └─────────┘
```

### Key Components

| Component | Description | Technology |
|-----------|-------------|------------|
| **Document Ingestion** | Load & chunk clinical PDFs/text files | PyPDF, LangChain TextSplitter |
| **Vector Store** | Semantic search over document chunks | FAISS + HuggingFace Embeddings |
| **Agentic Graph** | Multi-node reasoning pipeline | LangGraph StateGraph |
| **LLM** | Verification, contradiction detection, answer generation | Claude 3 Haiku (Anthropic) |
| **Evaluation** | Pipeline quality scoring | RAGAS + Custom Metrics |
| **UI** | Interactive web interface | Streamlit |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/nanda1045/Agentic-Research-Copilot-for-Clinical-Documents.git
cd Agentic-Research-Copilot-for-Clinical-Documents

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Generate Sample Data & Ingest

```bash
# Generate 50 synthetic clinical trial documents
uv run python main.py generate-data

# Ingest into FAISS vector store
uv run python main.py ingest
```

### 4. Query

```bash
# CLI query
uv run python main.py query "What are the side effects of Metformin in Type 2 Diabetes trials?"

# Or launch the web UI
uv run streamlit run app.py
```

### 5. Evaluate

```bash
# Generate eval dataset + run evaluation
uv run python main.py generate-eval
uv run python main.py evaluate

# With RAGAS metrics (uses API credits)
uv run python main.py evaluate --ragas
```

---

## 🧠 Agent Pipeline Details

### Node Descriptions

1. **Retrieve** — Embeds the query and searches FAISS for the top-k most similar document chunks. On multi-hop iterations, uses a reformulated query to fill evidence gaps.

2. **Verify** — LLM-as-judge evaluates each retrieved chunk for relevance to the query. Chunks below the verification threshold are discarded. Calculates initial confidence score.

3. **Contradict** — Pairwise comparison of verified chunks to detect conflicting claims (e.g., Drug X works vs. Drug X doesn't work). Flags contradictions with severity levels and adjusts confidence downward.

4. **Abstain** — Triggered when overall confidence falls below the threshold (default: 0.7). Returns a structured explanation of why the system cannot provide a reliable answer.

5. **Answer** — Generates a grounded response using only verified evidence, with inline citations. Acknowledges any contradictions for transparency. Includes confidence badge.

### Multi-Hop Reasoning

For complex questions requiring information from multiple documents:
- If the verify step finds insufficient evidence (<2 verified chunks), the agent **reformulates the query** and retrieves additional documents
- Maximum of 3 hops to prevent infinite loops
- Each hop targets specific evidence gaps identified in previous iterations

---

## 📊 Evaluation Metrics

### Custom Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Grounded Answer Accuracy** | % of answers fully supported by retrieved context | +28% over baseline RAG |
| **Unsupported Answer Rate** | % of answers containing unsupported claims | -31% over baseline RAG |
| **Abstention Precision** | % of correct abstentions (no good answer existed) | >80% |
| **Contradiction Detection Rate** | % of contradictory evidence correctly flagged | >70% |

### RAGAS Metrics (optional)

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Factual consistency of answer vs. context |
| **Answer Relevancy** | How well the answer addresses the question |
| **Context Precision** | Quality of retrieved context ranking |

---

## 📁 Project Structure

```
├── config/
│   └── settings.py              # Centralized configuration
├── data/
│   └── clinical_docs/           # Clinical document corpus
├── scripts/
│   └── generate_sample_data.py  # Synthetic data generator
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # PDF + text file loading
│   │   ├── chunker.py           # Recursive text splitting
│   │   └── vectorstore.py       # FAISS index management
│   ├── retrieval/
│   │   └── retriever.py         # Semantic similarity search
│   ├── agents/
│   │   ├── state.py             # LangGraph state definition
│   │   ├── nodes.py             # Agent nodes (retrieve, verify, contradict, abstain, answer)
│   │   └── graph.py             # LangGraph graph construction
│   └── evaluation/
│       ├── eval_data_generator.py  # Synthetic eval Q&A generator
│       └── evaluate.py          # Evaluation pipeline
├── app.py                       # Streamlit web UI
├── main.py                      # CLI entry point
├── pyproject.toml               # UV/pip project config
└── README.md
```

---

## 💡 Cost Optimization

This project is designed to minimize API costs:

| Component | Cost | Technology |
|-----------|------|------------|
| **Embeddings** | 🆓 Free | HuggingFace `all-MiniLM-L6-v2` (local) |
| **LLM** | 💰 Low | Claude 3 Haiku ($0.25/1M input tokens) |
| **Evaluation Data** | 🆓 Free | Template-based generation (no API) |
| **RAGAS** | 💰 Optional | Only runs when `--ragas` flag is used |

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Agent Framework:** LangGraph
- **LLM:** Anthropic Claude 3 Haiku
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Store:** FAISS
- **Document Processing:** PyPDF, LangChain
- **Evaluation:** RAGAS + Custom Metrics
- **UI:** Streamlit
- **Package Manager:** UV

---

## 📄 License

MIT License

## 👤 Author

Nanda Kishore Vuppili
