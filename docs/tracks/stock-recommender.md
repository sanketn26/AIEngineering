# Track: Phi / SLM Stock Recommender (90 days)

**Goal:** Production-minded stock *research assistant / recommender prototype* using classical ML baselines, retrieval over news, efficient open models, compression, and deployment.

!!! warning "Not financial advice"
    Educational project only. Markets are risky. Do not present outputs as personalized investment advice without proper licensing and disclaimers.

**Platform:** Any OS (macOS/Linux preferred); Windows via WSL2 is fine. VS Code/Cursor + GitHub.

**Updated for 2026:** Prefer current Phi / Llama / Qwen compact instruct models; use **model compression** vocabulary (not “MCP”); RAG with hybrid search; FastAPI + CI.

---

## Phase overview

| Phase | Days | Focus | Deliverable |
|-------|------|-------|-------------|
| Foundations | 1–14 | Python, Pandas, Git, data | Cleaned OHLCV + EDA notebook |
| Baseline ML | 15–28 | Trees/forests, backtest hygiene | Classifier + metrics README |
| SLM fine-tune | 29–42 | QLoRA / PEFT on financial text | Adapter + inference API |
| RAG & vectors | 43–56 | News/filings retrieval | Cited answers demo |
| Compression | 57–70 | Quantize / prune / distill | Size–quality report |
| Context engineering | 71–80 | Prompt templates, guards | Prompt pack + regression evals |
| Deploy & MLOps | 81–90 | Docker, CI, monitor | Deployed API + dashboard sketch |

**Core modules:** [01](../core/01-prompt-engineering.md)–[07](../core/07-tools-and-rag.md), [09](../core/09-advanced-rag.md), [10](../core/10-cost-optimization.md), [13](../core/13-production.md), [17](../core/17-small-models.md).

---

## Days 1–14 — Foundations

- Environment: Python 3.11+, Poetry or venv, pre-commit optional  
- Data: [yfinance](https://pypi.org/project/yfinance/), Kaggle market datasets  
- Skills: [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)  
- Build cleaning + feature pipeline under `src/` + `notebooks/`  
- Glossary of OHLCV / corporate-action pitfalls ([Investopedia OHLCV](https://www.investopedia.com/terms/o/open-high-low-close-volume-ohlcv.asp))  

**Exit:** Reproducible data pull script; EDA plots; repo structure.

---

## Days 15–28 — Classical baseline

- Train/val/test with **time-aware** splits (no random shuffle leakage)  
- Models: logistic regression / random forest baselines ([sklearn](https://scikit-learn.org/stable/user_guide.html))  
- Metrics: precision/recall carefully interpreted; simple backtest with costs assumptions documented  
- Optional tuning: Optuna  

**Exit:** `models/baseline` + metrics report; honesty about limitations.

---

## Days 29–42 — Efficient open LLM

- Pick a small instruct model (Phi-4-class / Llama compact / Qwen) on Hugging Face  
- Task: headline or filing → structured label / short rationale  
- PEFT LoRA/QLoRA ([PEFT docs](https://huggingface.co/docs/peft/index), [QLoRA paper](https://arxiv.org/abs/2305.14314))  
- Serve `POST /predict` with FastAPI  

**Exit:** Adapter weights (or training script) + API README.

---

## Days 43–56 — RAG

- Index news or filings (respect licenses/ToS)  
- Vector store: FAISS / Chroma / Qdrant / Pinecone  
- Pipeline: retrieve → generate with citations  
- Eval: Hit@k on a hand-labeled question set  

**Exit:** Demo notebook or UI returning **cited** rationales.

---

## Days 57–70 — Compression

- Quantization (GPTQ/AWQ/GGUF as appropriate)  
- Optional: pruning / distillation experiments  
- Measure latency, VRAM, and golden-set quality  

**Exit:** Compression report + `--lite-model` flag or separate artifact.

---

## Days 71–80 — Context & prompts

- Template pack for: screen → explain → risk bullets  
- Injection-resistant system policy (Module 02)  
- Regression evals on fixed prompts (Module 04)  

**Exit:** Versioned `prompts/` + eval JSON summary.

---

## Days 81–90 — Deploy

- Docker image; GitHub Actions CI (lint, tests, eval subset)  
- Config for CPU vs GPU  
- Minimal metrics (latency, error rate, token/$ if applicable)  
- Grafana or simpler dashboard optional  

**Exit:** Deployed or clearly runnable compose stack + screencast.

---

## Milestone table

| Day | Checkpoint |
|-----|------------|
| 14 | Data pipeline + EDA |
| 28 | Baseline ML + report |
| 42 | Fine-tuned SLM API |
| 56 | RAG with citations |
| 70 | Compressed model report |
| 80 | Prompt/eval pack |
| 90 | Deploy + demo |

---

## Resource index (curated)

| Topic | Link |
|-------|------|
| WSL (Windows) | https://learn.microsoft.com/en-us/windows/wsl/install |
| VS Code WSL | https://code.visualstudio.com/docs/remote/wsl |
| FastAPI | https://fastapi.tiangolo.com/tutorial/ |
| GitHub Actions | https://docs.github.com/en/actions/quickstart |
| FAISS | https://github.com/facebookresearch/faiss |
| LangChain RAG (optional) | https://python.langchain.com/docs/ |

---

## Success checklist

- [ ] Time-safe ML baseline documented  
- [ ] SLM path reproducible  
- [ ] RAG cites sources  
- [ ] Compression tradeoffs measured  
- [ ] CI + container runbook  
- [ ] Clear non-advice disclaimer in UX/README  
