# Prerequisites & Setup

## What you need

**Skills**

- Python 3.11+ at intermediate level (functions, classes, async basics)
- Comfort with HTTP APIs and JSON
- Git basics; a GitHub account
- Optional: Docker, basic Linux CLI

**Hardware**

| Path | Minimum | Comfortable |
|------|---------|-------------|
| API-only (cloud models) | 8 GB RAM laptop | 16 GB |
| Local 7B–8B models | 16 GB RAM | 32 GB + GPU (8 GB+ VRAM) |
| Fine-tuning / large RAG | 16 GB + GPU | 24 GB+ VRAM |

**Accounts (pick what you need)**

- At least one LLM provider: [OpenAI](https://platform.openai.com/), [Anthropic](https://www.anthropic.com/), [Google AI](https://ai.google.dev/), or free local via [Ollama](https://ollama.com/)
- Optional: Hugging Face, Pinecone/Weaviate/Qdrant, cloud host (Fly, Railway, AWS, Azure, GCP)

---

## Environment setup

### 1. Clone and Python env

**Python:** 3.11–3.13 recommended (`^3.11,<3.14` in `pyproject.toml`). Core teaching modules are **stdlib-only**.

```bash
git clone https://github.com/<you>/AIEngineering.git
cd AIEngineering

# Poetry (repo default)
poetry config virtualenvs.in-project true
poetry env use python3.11   # if multiple Pythons installed
poetry install --with dev
# Optional stock/data track extras (pandas, numpy, sklearn, …):
# poetry install -E track-data

# Or venv + pip (core tests only)
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip pytest
PYTHONPATH=. pytest tests/ -m "not track_data"
```

### 2. Core learning packages

Install as you need modules (not everything on day 1):

```bash
pip install openai anthropic python-dotenv httpx tenacity tiktoken
pip install pydantic pydantic-settings
# RAG
pip install numpy scikit-learn  # baselines / embeddings helpers
# Optional later
# pip install chromadb sentence-transformers fastapi uvicorn
# pip install peft transformers accelerate bitsandbytes  # fine-tune
```

### 3. Secrets

Create `.env` (never commit it):

```bash
# .env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
# Optional
HF_TOKEN=
```

Load with `python-dotenv` or your shell.

### 4. Smoke test (provider-agnostic idea)

```python
"""setup_smoke.py — adapt to your provider."""
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Prefer Anthropic or OpenAI if keys exist; else print local guidance
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16,
        )
        print(r.choices[0].message.content)
        return
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        )
        print(r.content[0].text)
        return
    print("No API key found. Install Ollama and run: ollama run llama3.2")

if __name__ == "__main__":
    main()
```

### 5. Local models (optional, recommended)

```bash
# macOS / Linux / Windows
# https://ollama.com
ollama pull llama3.2
ollama run llama3.2
```

---

## Course sandbox package

This repo includes runnable teaching code under `src/`:

| Module | Purpose |
|--------|---------|
| `src.security` | Sanitization + PII redaction |
| `src.prompts` | Template render helpers |
| `src.context_memory` | Session memory + token budget |
| `src.rag` | TinyRAG + RRF |
| `src.evals` | Golden-set runner |
| `src.cost` | Router, cache, spend ledger |
| `src.agents` | Single-agent loop |
| `src.audit` | Hashing audit events |

```bash
poetry install
poetry run pytest tests/ -v
```

Guided exercises: [Exercises](../reference/exercises.md) · grading: [Assessment](../reference/assessment.md).

### Recommended layout for *your* track work

```text
your-work/
  prompts/
  notebooks/          # exploration only
  src/                # or extend this repo's src/
  tests/
  evals/              # golden sets, prompt fixtures
  .env
  PROGRESS.md         # daily log
```

---

## Windows notes

- **WSL2** is strongly recommended for path parity with tutorials: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- Open the WSL folder in VS Code with the Remote – WSL extension
- Keep Git line endings consistent (`core.autocrlf`)

macOS / Linux: native terminal is fine; VS Code or Cursor optional.

---

## Docs site (contributors)

```bash
pip install -r requirements-docs.txt
mkdocs serve   # http://127.0.0.1:8000
mkdocs build   # site/ for static export
```

---

## Next

- Choose a [learning path](paths.md)
- Start [Module 01 — Prompt engineering](../core/01-prompt-engineering.md)
