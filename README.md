# AI Engineering Course

**From first prompt to production agents** — a restructured, relevance-updated curriculum with optional 90-day specialization tracks.

This repository is optimized to publish as a **GitHub Pages** site via [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

| | |
|---|---|
| **Docs (local)** | `mkdocs serve` → http://127.0.0.1:8000 |
| **Curriculum source** | [`docs/`](docs/) |
| **Progress / XP** | [docs/getting-started/progress.md](docs/getting-started/progress.md) (localStorage gamification) |
| **Review of old materials** | [docs/review/analysis.md](docs/review/analysis.md) |
| **Originals (archived)** | [`archive/source/`](archive/source/) |
| **Python sandbox** | [`src/`](src/) + Poetry (`pyproject.toml`) |

---

## What changed in the restructure

| Before | After |
|--------|--------|
| 11k-line scrambled LLM guide | 17 ordered core modules |
| Three disconnected 90-day plans | Tracks linked to shared core |
| “MCP” misused (load balancer / compression) | Correct Model Context Protocol + clear compression naming |
| gpt-3.5-centric examples | Provider-agnostic, 2026-oriented stack |
| Flat root markdown | MkDocs Material + GitHub Actions Pages |
| Outline-style modules | CS-engineer tutorials (explainers, quizzes, labs, diagrams) |
| No progress UX | Static gamification (XP, badges, HUD) via `localStorage` |

Full diagnostic: **[Review analysis](docs/review/analysis.md)**.

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

Runnable teaching modules: security, prompts, context memory, RAG, evals, cost, agents, audit.

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

## Curriculum map

**Core modules:** prompting → security → advanced prompts → evals → context engineering → fine-tuning → tools/RAG → MCP → advanced RAG → cost → agents → multi-agent → production → compliance → domains → integration → small/local models.

**Tracks (90 days):**

1. [Stock recommender (SLM + RAG + MLOps)](docs/tracks/stock-recommender.md)  
2. [Hybrid Transformer+MLP from scratch](docs/tracks/hybrid-models.md)  
3. [Agentic VS Code plugin](docs/tracks/agentic-plugin.md)  

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
  core/                    # Modules 01–17
  tracks/                  # 90-day specializations
  reference/
  review/analysis.md
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
