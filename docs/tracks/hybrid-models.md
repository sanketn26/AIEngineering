# Track: Hybrid Transformer + MLP from Scratch (90 days)

**Goal:** Design, train, evaluate, and ship a **custom hybrid** architecture fusing Transformer and MLP blocks for a use case you choose (tabular+text, time series, or structured+sequence data).

**Tools:** Python, PyTorch, VS Code/Jupyter, GitHub.

**Core modules:** [05](../core/05-context-engineering.md)–[06](../core/06-fine-tuning.md) (concepts transfer); this track is primarily deep learning engineering.

---

## Phase overview

| Phase | Days | Focus | Deliverable |
|-------|------|-------|-------------|
| Foundations | 1–14 | Env, data, use case, DL basics | Clean env + data card |
| Prototypes | 15–28 | Standalone MLP & Transformer | Two working training scripts |
| Hybrid design | 29–42 | Fusion architecture | Modular hybrid `nn.Module` |
| Train & explain | 43–56 | Loops, tuning, interpretability | Baseline comparison report |
| Optimize | 57–70 | Scaling, regularization, ablations | Final architecture choice |
| Deploy | 71–84 | Export, API/CLI, tests | Served or CLI model |
| Publish | 85–90 | Docs, polish, write-up | Public repo + report |

---

## Days 1–14 — Foundations

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or venv  
- [PyTorch install](https://pytorch.org/get-started/locally/)  
- Background: [How transformers work](https://www.datacamp.com/tutorial/how-transformers-work), [MLP vs Transformer](https://www.exxactcorp.com/blog/deep-learning/when-to-use-mlps-vs-transformers)  
- Select dataset ([Kaggle](https://www.kaggle.com/datasets) or your own) with **mixed** feature types if possible  
- Data card: sizes, splits, leakage risks, metrics  

---

## Days 15–28 — Prototypes

- MLP: [Neural nets with PyTorch](https://machinelearningmastery.com/neural-networks-with-pytorch/)  
- Transformer: [torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html), [UVA DL notebook](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)  
- Data: [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)  
- Logging: [TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)  

**Exit:** Separate trainable MLP and Transformer baselines on your data.

---

## Days 29–42 — Hybrid design

Study fusion ideas (examples; verify papers):

- MLP-Mixer lineage: [arXiv:2105.01601](https://arxiv.org/abs/2105.01601)  
- Time series MLP mixers (e.g. TSMixer on Papers with Code)  
- Domain hybrid papers relevant to *your* modality  

Implement:

- Independent `MLPBlock` and `TransformerBlock` modules  
- Fusion options: concat embeddings, gated sum, cross-attention, wide-and-deep  
- Config-driven architecture (YAML/JSON) for ablations  

---

## Days 43–56 — Train, tune, explain

- Training loops & schedulers ([PyTorch optim](https://pytorch.org/docs/stable/optim.html))  
- Early stopping / checkpointing  
- Compare hybrid vs MLP-only vs Transformer-only  
- Interpretability: [Captum](https://captum.ai/), attention visualizations where meaningful  

**Exit:** Results table + error analysis.

---

## Days 57–70 — Optimize

- Depth/width sweeps, dropout, weight decay, norm choices  
- Efficiency: mixed precision, torch.compile (where stable)  
- Task-specific heads and losses  

**Exit:** Frozen architecture + training recipe.

---

## Days 71–84 — Deploy

- `torch.save` / [ONNX export](https://onnxruntime.ai/docs/export/models.html)  
- FastAPI or Streamlit demo  
- Unit tests for preprocessing and shapes  
- Document limits and intended use  

---

## Days 85–90 — Publish

- Clean README, architecture diagram, reproduction steps  
- Short report or blog-style write-up  
- Tag `v0.1.0` release  

---

## Milestones

| Day | Checkpoint |
|-----|------------|
| 14 | Env + data card |
| 28 | Two baselines train |
| 42 | Hybrid module complete |
| 56 | Comparison report |
| 70 | Optimized recipe |
| 84 | Demo API/CLI |
| 90 | Public polished repo |

---

## Success checklist

- [ ] Hybrid design justified vs. baselines  
- [ ] Reproducible training command  
- [ ] Ablations documented  
- [ ] Export path works on clean machine  
- [ ] Honest failure analysis  
