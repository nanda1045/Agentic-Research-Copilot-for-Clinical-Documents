.PHONY: help setup data ingest run query eval clean

# Default target
help: ## Show this help message
	@echo ""
	@echo "  Agentic Research Copilot"
	@echo "  ========================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## Install all dependencies (requires uv)
	@command -v uv >/dev/null 2>&1 || { echo "  Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	uv sync
	@echo "  ✓ Dependencies installed"
	@test -f .env || (cp .env.example .env && echo "  → Created .env from template. Add your ANTHROPIC_API_KEY.")

setup-eval: ## Install with evaluation extras (RAGAS)
	uv sync --extra eval
	@echo "  ✓ Dependencies + evaluation extras installed"

data: ## Generate 50 synthetic clinical documents
	uv run python main.py generate-data

ingest: ## Ingest documents into FAISS vector store
	uv run python main.py ingest

run: ## Launch the Streamlit web UI
	uv run streamlit run app.py

query: ## Run a CLI query (usage: make query Q="your question")
	uv run python main.py query "$(Q)"

eval: ## Run evaluation pipeline
	uv run python main.py generate-eval
	uv run python main.py evaluate

eval-ragas: ## Run evaluation with RAGAS metrics (uses API credits)
	uv run python main.py evaluate --ragas

clean: ## Remove generated data, index, and eval results
	rm -rf faiss_index/*
	rm -rf eval_results/*
	rm -rf data/clinical_docs/*.txt
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	@echo "  ✓ Cleaned generated files"

lint: ## Run linter (requires dev extras)
	uv run ruff check .

format: ## Auto-format code (requires dev extras)
	uv run ruff format .
