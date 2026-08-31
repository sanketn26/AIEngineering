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

## Curriculum map

**Core modules:** prompting → security → advanced prompts → evals → context engineering → fine-tuning → tools/RAG → MCP → advanced RAG → cost → agents → multi-agent → production → compliance → domains → integration → small/local models → agent design patterns → orchestration patterns → **reliability → secure tool use → harness engineering → agent evals → prompt drift → local-first agents → durable orchestration → orchestrator comparison**.

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
  core/                    # Modules 01–27
  tracks/                  # 90-day specializations
  reference/
archive/source/            # Pre-restructure markdown (provenance)
src/ tests/                # Optional Poetry project
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
