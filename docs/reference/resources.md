# Curated Resources

Prefer primary documentation and high-signal open-source curricula over random blogs. Links are entry points — always check for updates.

---

## Open-source curricula (start here)

| Resource | Best for | How to use with this course |
|----------|----------|------------------------------|
| [DAIR.AI Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) · [promptingguide.ai](https://www.promptingguide.ai/) | Prompts, CoT, ReAct, agents overview | Modules **01–03**, skim before building agents |
| [mlabonne/llm-course](https://github.com/mlabonne/llm-course) | LLM engineer track (RAG, deploy, quant) | Modules **06–07, 09, 13, 17** — take “Engineer” not full “Scientist” unless training is your job |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Attention, GPT training loop | Hybrid track + mental model for **06 / 17** |
| [huggingface/agents-course](https://github.com/huggingface/agents-course) | smolagents, LangGraph, agentic RAG | Modules **11–12**, **26**, agentic pieces of **09** |
| [humanlayer/12-factor-agents](https://github.com/humanlayer/12-factor-agents) | Production agent design principles | Modules **05, 11–13, 20–25** |
| [Anthropic / MCP](https://modelcontextprotocol.io/) | Tools, resources, prompts protocol | Module **08** (authoritative) + host policy in **21** |

---

## Models & providers

- [OpenAI docs](https://platform.openai.com/docs)
- [Anthropic docs](https://docs.anthropic.com/)
- [Google AI Gemini](https://ai.google.dev/)
- [Hugging Face Hub](https://huggingface.co/models)
- [Ollama](https://ollama.com)

---

## RAG & embeddings

- [FAISS](https://github.com/facebookresearch/faiss)
- [Chroma](https://docs.trychroma.com/)
- [Qdrant](https://qdrant.tech/documentation/)
- [Pinecone docs](https://docs.pinecone.io/)
- Sentence-transformers / provider embedding guides
- LlamaIndex & LangChain retrieval docs (orchestration patterns)

---

## Agents & orchestration

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
- [AutoGen / AG2](https://github.com/microsoft/autogen)
- Provider agent / tool-calling guides (OpenAI, Anthropic)
- Course **Module 18** (leaf patterns), **19** (workflow shape), **20–26** (reliability, sandbox, evals, drift, durable graphs, orchestrator comparison) — read 11–12 first
- [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) · [ReWOO (Xu et al., 2023)](https://arxiv.org/abs/2305.18323) · [Self-consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)

---

## MCP

- [Model Context Protocol](https://modelcontextprotocol.io/specification/2026-07-28) — **not** a load balancer; tools/resources/prompts over a standard host↔server protocol. Course teaches **2026-07-28** (stateless `_meta` + `server/discover`). The 2025 `initialize` + `Mcp-Session-Id` model is historical and still deployed.

---

## Fine-tuning

- [PEFT](https://huggingface.co/docs/peft/index)
- [TRL](https://huggingface.co/docs/trl)
- QLoRA paper: https://arxiv.org/abs/2305.14314

---

## Evals & observability

- [promptfoo](https://github.com/promptfoo/promptfoo)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [RAGAS](https://github.com/explodinggradients/ragas)
- Langfuse, Arize Phoenix, OpenTelemetry

---

## Security

- OWASP LLM Top 10 (search current version)
- Prompt injection write-ups from major vendors (treat as living docs)
- Microsoft Presidio (PII detection) for production redaction

---

## MLOps & serving

- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- vLLM / Text Generation Inference docs

---

## Deep learning (hybrid track)

- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [PyTorch nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

---

## Classic data / ML

- [Pandas 10 minutes](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
- [yfinance](https://pypi.org/project/yfinance/)

---

## Course provenance

Pre-restructure originals: `archive/source/` in this repository.  
Progress (local XP): [Progress & gamification](../getting-started/progress.md).
