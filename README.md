# AI Engineering Course

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Buy%20me%20a%20coffee-FFDD00?style=flat-square)](https://buymeacoffee.com/sanketn)

**From first prompt to production agents** — a restructured, relevance-updated curriculum with optional 90-day specialization tracks. Core now includes **production-agent** modules (reliability, sandboxing, trajectory evals, drift, local-first, durable orchestration, orchestrator trade-offs).

The published course is available at **[sanketn26.github.io/AIEngineering](https://sanketn26.github.io/AIEngineering/)** and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

| | |
|---|---|
| **Docs (local)** | `mkdocs serve` → http://127.0.0.1:8000 |
| **Curriculum source** | [`docs/`](docs/) |
| **Progress / XP** | [docs/getting-started/progress.md](docs/getting-started/progress.md) (localStorage gamification) |
| **Originals (archived)** | [`archive/`](archive/) — provenance only, not the curriculum |
| **Python sandbox** | [`src/`](src/) + Poetry (`pyproject.toml`) |
| **Capstone starter** | [`capstone-starter/`](capstone-starter/) — triage service; mock model; five gates |
| **Command-runtime capstone** | [`capstone-command/`](capstone-command/) — a 4B model may propose a patch; the runtime decides |
| **Divide, solve, and join** | [`capstone-decompose/`](capstone-decompose/) — a 20B one-shot fails; dividing and joining can finish the task |
| **Track starters** | [`tracks/starters/`](tracks/starters/) — one vertical slice per track, not the 90-day solution |

---

## What changed in the restructure

| Before | After |
|--------|--------|
| 11k-line scrambled LLM guide | 27 ordered core modules |
| Three disconnected 90-day plans | Tracks linked to shared core |
| “MCP” misused (load balancer / compression) | Correct Model Context Protocol + clear compression naming |
| gpt-3.5-centric examples | Provider-agnostic, 2026-oriented stack |
| Flat root markdown | MkDocs Material + GitHub Actions Pages |
| Outline-style modules | CS-engineer tutorials (explainers, quizzes, labs, diagrams) |
| No progress UX | Static gamification (XP, badges, HUD) via `localStorage` |

---

## Quick start — read the course

```bash
# Docs-only dependencies
python3 -m venv .venv-docs
source .venv-docs/bin/activate   # Windows: .venv-docs\Scripts\activate
pip install -r requirements-docs.txt
mkdocs serve
```

Open the printed local URL. Navigation: Home → Getting started → Core modules → Tracks.

---

## Quick start — Python sandbox

Runnable teaching modules: security, prompts, context memory, RAG, evals, cost, agents, audit, reliability, sandbox, harness, agent evals, MCP prod, drift, local-first agents, durable orchestration, orchestrator comparison.

```bash
# Python 3.11–3.13 (not 3.14 yet for optional scientific stack)
poetry config virtualenvs.in-project true
poetry env use python3.11   # recommended
poetry install --with dev
make test                   # core: security, rag, agents, evals, …

# Stock / data track extras (pandas, numpy, sklearn, matplotlib, yfinance)
poetry install -E track-data
poetry run pytest tests/ -v
```

Exercises: [docs/reference/exercises.md](docs/reference/exercises.md) · Rubrics: [docs/reference/assessment.md](docs/reference/assessment.md)

**Fine-tuning lab:** the [QLoRA walkthrough](docs/core/06-qlora.md) for a 1.5B model — rights gate, cleaning, numeric-grounding checks, and company-level splits in stdlib `src/finetune_data.py`. Training and base-vs-adapter eval in [`examples/fine-tuning/`](examples/fine-tuning/README.md) are the optional GPU half.

**Inference lab:** [Module 28](docs/core/28-inference-serving.md) and its [experiments](docs/core/inference/hands-on.md) compare KV caching, attention backends, and speculative decoding, with GPU extensions for prefix reuse and batching. Start with `--smoke` for real forward passes without downloading weights; install its separate dependencies from [`examples/inference/`](examples/inference/README.md).

---

## Quick start — capstone starter

Independently of Poetry/`src/`. No API keys.

```bash
cd capstone-starter
pip install -r requirements.txt
uvicorn app:app --reload
pytest tests/ -v
```

Gates: [docs/core/capstone-gates.md](docs/core/capstone-gates.md) · spec: [docs/core/capstone.md](docs/core/capstone.md)

---

## Quick start — command-runtime capstone

A second capstone. A mock diff stands in for a 4B code model. No API keys.

```bash
cd capstone-command
pip install -r requirements.txt
python cmdai.py spec check fixtures/retry_api_client.yaml
python cmdai.py run --intent "Add retry support"
pytest tests/ -v
```

Spec: [docs/core/capstone-command.md](docs/core/capstone-command.md) · gates: [docs/core/capstone-command-gates.md](docs/core/capstone-command-gates.md)

---

## Quick start — divide, solve, and join

A third capstone. One prompt to a 20B model fails. The starter still treats that prompt as success. No API keys.

```bash
cd capstone-decompose
pip install -r requirements.txt
python divide.py check fixtures/refund_task.yaml
python divide.py run fixtures/refund_task.yaml
pytest tests/ -v
```

Spec: [docs/core/capstone-decompose.md](docs/core/capstone-decompose.md) · gates: [docs/core/capstone-decompose-gates.md](docs/core/capstone-decompose-gates.md)

---

## Follow the experiments

A higher score, a real citation, and a saved approval can each hide a failure.
These labs let you catch it:

- [Did the score really improve?](docs/core/04-testing-evals.md#the-score-moved-is-that-enough) — paired cases and an uncertainty-aware release gate, inside Module 04.
- [The citation was real. The answer was wrong.](docs/core/09-advanced-rag.md#compare-the-paths-on-one-corpus) — four retrieval paths on one labeled corpus, inside Module 09.
- [The room had a sign, but no lock.](docs/core/21-secure-tool-use.md#6-a-boundary-the-operating-system-can-enforce) — filesystem and network probes, inside Module 21.
- [A refund is suggested. Who gets to say yes?](docs/core/reference-capstone.md) — a completed five-gate service alongside the student starter.
- `python -m examples.durability.crash` — interrupt a workflow after the effect and before its receipt; explain why the retry stays safe.

## Curriculum map

**Core modules:** prompting → security → advanced prompts → evals → context engineering → fine-tuning → tools/RAG → MCP → advanced RAG → cost → agents → multi-agent → production → compliance → domains → integration → small/local models → **inference serving** → agent design patterns → orchestration patterns → **reliability → secure tool use → harness engineering → agent evals → prompt drift → local-first agents → durable orchestration → orchestrator comparison**.

**Tracks (90 days):** start from the slice, not a blank repo.

1. [Stock recommender](docs/tracks/stock-recommender.md) — [`tracks/starters/stock-recommender/`](tracks/starters/stock-recommender/)
2. [Hybrid Transformer+MLP](docs/tracks/hybrid-models.md) — [`tracks/starters/hybrid-models/`](tracks/starters/hybrid-models/)
3. [Agentic VS Code plugin](docs/tracks/agentic-plugin.md) — [`tracks/starters/agentic-plugin/`](tracks/starters/agentic-plugin/)  

---

## Enable GitHub Pages

1. Push this repo to GitHub.  
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**  
3. Push to `main` (or run the **Deploy docs to GitHub Pages** workflow manually).  
4. Site URL will be `https://<user>.github.io/<repo>/` for project sites.

The workflow is [`.github/workflows/deploy-docs.yml`](.github/workflows/deploy-docs.yml). It sets `site_url` automatically during CI.

---

## Repository layout

```text
docs/                      # Course site (source of truth)
  getting-started/
  core/                    # Modules 01–28, plus three capstones
  tracks/                  # 90-day specializations
  reference/
archive/source/            # Pre-restructure markdown (provenance)
src/ tests/                # Optional Poetry project
capstone-starter/          # Triage service with five planted holes
capstone-command/          # Specification-gated coding runtime
capstone-decompose/       # 20B divide, solve, and join
mkdocs.yml
requirements-docs.txt
.github/workflows/deploy-docs.yml
```

---

## Contributing

- Prefer editing `docs/**` over reintroducing monolith files at the repo root.  
- Run `mkdocs build --strict` before opening a PR that touches docs.  
- Keep examples short and honest; mark domain/legal content as non-advice.

---

## License

See [LICENSE](LICENSE).
