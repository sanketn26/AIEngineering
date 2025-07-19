# 90-Day Phi-Based Stock Recommender Roadmap  
**Platform:** Windows laptop + WSL, VS Code, GitHub  
**Goal:** End-to-end, production-grade stock recommendation system with RAG, vector DB, model compression (MCP), and context engineering—using small, efficient open models.

---

## 📅 Phase & Milestones Overview

| Phase                   | Days  | Key Skills/Topics                        | Deliverables                                  |
|-------------------------|-------|------------------------------------------|------------------------------------------------|
| Foundations             | 1-14  | Python, Pandas, Git, VS Code, WSL setup  | Cleaned data, exploratory notebook             |
| Baseline ML             | 15-28 | Decision/Random Forests, evaluation      | ML classifier w/ validation report, README     |
| Phi Fine-Tuning         | 29-42 | QLoRA, PEFT, open LLMs (Phi)             | Fine-tuned Phi model, REST API                 |
| Retrieval (RAG) & Vector DB | 43-56 | RAG, FAISS/Pinecone, LangChain           | RAG demo: cited answers with vector search     |
| Compression & Pruning   | 57-70 | MCP (prune, quantize, distil), TorchAO   | Compressed model, CI job, compression report   |
| Context Engineering     | 71-80 | Prompt tactics, retrieval orchestration   | Deterministic, robust prompts/templates        |
| Deployment & MLOps      | 81-90 | Docker, FastAPI, CI/CD, retraining, dashboards | Deployed API with retrain pipeline & dashboard  |

---

## 🔗 Essential Resources

| Topic                      | Resource/Link                                                                                              |
|----------------------------|------------------------------------------------------------------------------------------------------------|
| Python, VS Code, WSL       | [Set up WSL on Windows][1], [VS Code WSL Guide][2], [GitHub Hello World][3]                               |
| Data Wrangling             | [10 Minutes to Pandas][4], [YFinance for Financial Data][5]                                               |
| Financial Data Primer      | [Investopedia: OHLCV Explained][6], [Kaggle: Stock Market Datasets][7]                                     |
| Git & Project Management   | [Git & GitHub Crash Course][8], [Pre-commit Hooks Guide][9]                                               |
| ML Foundation (Scikit-Learn)| [Scikit-learn User Guide][10], [KDnuggets: ML Evaluation Metrics][11]                                     |
| Open LLMs & Phi            | [Microsoft Phi-3 Github][12], [HuggingFace Phi-3 quickstart][13], [PEFT Intro Blog][14], [QLoRA Paper][15]|
| RAG + Vector DB            | [DeepLearning.ai RAG Specialization][16], [LangChain RAG Docs][17], [Pinecone Vector DB Docs][18], [FAISS Tutorial][19]       |
| Model Compression (MCP)    | [Datature Model Pruning Guide][20], [TorchAO Pruning Tutorial][21], [Knowledge Distillation Doc][22], [SparseGPT Example][23]|
| Context Engineering        | [DeepLearning.ai Prompt Engineering][24], [LangChain Advanced Prompting][25]                               |
| MLOps                      | [Dockerizing ML Apps][26], [GitHub Actions CI/CD Guide][27], [FastAPI Tutorial][28], [Grafana Dashboard][29]|

---

## 🖥️ Windows + VS Code + GitHub Setup (Days 1–2)

- **Install WSL 2:**  
  [Microsoft WSL Install][1]  
- **VS Code Remote Integration:**  
  [Run VS Code in WSL][2]  
- **Install core packages:**  

``` bash
sudo apt update
sudo apt install python3 python3-pip git
pip install pandas numpy scikit-learn matplotlib yfinance
```

- **GitHub setup:**  
[First repo guide][3]

---

## 📊 Data Exploration and Preprocessing (Days 3–14)

- Use [10 Minutes to Pandas][4] for core DataFrame skills.
- Load S&P historical and Kaggle price data ([YFinance][5], [Kaggle][7]).
- Build a cleaning/feature pipeline in a `notebooks/` and modular Python file.
- Visualize with matplotlib.
- Write a glossary from [Investopedia][6] terms.
- Add pre-commit git hooks ([Guide][9]).

---

## 🌲 Classical ML Baseline (Days 15–28)

- **Train/test split & cross-validation:** Follow the [Scikit-learn Guide][10].
- **Random Forests:** Tune with [Optuna][11].
- **Backtesting:** Hold out most recent years.
- **Evaluation:** Visualize metrics, confusion matrices ([KDnuggets][11]).
- **Wrap up** in `models/` with a README summary.

---

## 🤖 Fine-Tuning Phi with QLoRA & PEFT (Days 29–42)

- Learn via [PEFT Intro][14] and [QLoRA Paper][15].
- Clone [Microsoft’s Phi-3 repo][12], run base inference ([Phi HuggingFace][13]).
- Prepare a Financial Text Dataset:  
- For realistic prototype, pair headlines with labels (simulate/scrape via yfinance, news-API, or use [Kaggle financial text sets][7]).
- Implement LoRA adapters, fine-tune for one epoch, monitor validation ([Hugging Face PEFT][14]).
- Save weights, write an inference REST endpoint.
- Optional: Study [movement pruning abstract][21] for forward compatibility.

---

## 🔍 RAG & Vector Database (Days 43–56)

- **Watch/read:**  
- [DeepLearning.ai RAG Specialization, Modules 1–3][16]
- [LangChain RAG docs and tutorials][17]
- **Build:**  
- Index news/articles with [FAISS][19]
- Try Pinecone for scalable storage ([docs][18])
- RAG chain: retrieve→generate with Phi ([tutorial][17]).
- Add retrieval-backed explainability to recommendations (cited sources).

---

## 📉 Model Compression & Pruning (Days 57–70)

- **Learn theory:**  
- [Datature Pruning Guide][20]
- [TorchAO Pruning Example][21]
- [SparseGPT Demo][23]
- **Apply:**  
- Try semi-structured pruning on Phi adapters.
- Quantize with GPTQ to reduce footprint.
- Assess loss vs gain with real metrics.
- **Write up:**  
- Compression report and add `--lite-model` CLI flag.

---

## 🧠 Context Engineering (Days 71–80)

- **Watch/read:**  
- [Prompt Engineering: DeepLearning.ai][24]
- [LangChain advanced patterns][25]
- **Build:**  
- Robust and templated prompts (“thought→tool→answer”).
- Manage retrieval context, context windows, and guardrails.
- Test on a suite of queries; write a context engineering handbook.

---

## 🚀 Production & MLOps (Days 81–90)

- **Dockerize** your API ([Guide][26]), add support for both GPU/CPU.
- **FastAPI** for serving and prediction.
- **CI/CD:**  
- Set up with [GitHub Actions][27].
- Lint, test, deploy images.
- **Automation:**  
- Cron job for retrain ([docs][27]).
- Backtest and log Sharpe ratios, portfolio metrics.
- **Monitoring:**  
- Grafana dashboard for usage, latency, accuracy ([Guide][29]).
- **Deliverables:**  
- Complete README, screencast demo, file logs of portfolio simulation.

---

## 📝 Daily Learning Schedule (1 hour/day)

1. **Review:** 10–15 min recap of previous material.
2. **New Content:** 30–35 min: either guided article/video or implementation.
3. **Hands-On:** 15–20 min: code notebooks, pipeline building, test suite.
4. **Project Log:** 5 min: update a `PROGRESS.md` in your repo.

---

## 🎓 Further Reading and Formal Path for Each Area

### Foundations:  
- [Set up WSL on Windows][1]
- [VS Code for WSL][2]

### Data Science & Visualization:
- [10 Minutes to Pandas][4]
- [YFinance usage][5]
- [Investopedia: OHLCV][6]

### ML & Baselines:
- [Scikit-learn User Guide][10]
- [KDnuggets on Model Metrics][11]

### Open LLM & Phi:
- [Microsoft Phi-3 GitHub][12]
- [HuggingFace Quickstart][13]
- [PEFT Blog][14]
- [QLoRA Paper][15]

### Retrieval & Vector DB:
- [DeepLearning.ai RAG][16]
- [LangChain Docs][17]
- [Pinecone Docs][18]
- [FAISS Tutorial][19]

### Model Compression:
- [Datature Pruning][20]
- [TorchAO Prune][21]
- [Intel KD Docs][22]
- [SparseGPT Example][23]

### Context Engineering:
- [Prompt Engineering: DeepLearning.ai][24]
- [LangChain Advanced Prompting][25]

### MLOps/Deployment:
- [Dockerizing ML Apps][26]
- [GitHub Actions CI/CD][27]
- [FastAPI Tutorial][28]
- [Grafana Setup][29]

---

## 🚩 Tips for WSL, VS Code, and GitHub

- Use [VS Code Remote][2] to seamlessly code inside WSL/Ubuntu from Windows.
- Keep all your project work (source, notebooks, report) version-controlled in GitHub ([Git and GitHub Crash Course][8]).
- Store daily notes and checkpoint progress in `PROGRESS.md`—use pre-commit hooks to enforce code formatting ([Pre-commit Guide][9]).

---

## 📈 Progress Milestones Table

| Day  | Deliverable                                   |
|------|-----------------------------------------------|
| 14   | Data pipeline, feature notebook, repo ready   |
| 28   | ML baseline (Decision Tree/Random Forest), README |
| 42   | Fine-tuned Phi model, basic inference API     |
| 56   | RAG vector search, news-backed recommendation |
| 70   | Pruned/compressed model, detailed report      |
| 80   | Prompt/context audit, regression test suite   |
| 90   | Dockerised, CI/CD deployed, demo + dashboard  |

---

## 🏁 Checklist for Success

- [ ] WSL, VS Code, Python/conda environment ready
- [ ] GitHub repo with modular project structure
- [ ] ML baseline model with solid metrics
- [ ] Fine-tuned, compressed Phi-based model
- [ ] Working RAG chain with vector DB, cited rationale
- [ ] Context engineering templates/regression suite
- [ ] CI/CD: Testing, deployment, retraining
- [ ] Dashboard and documentation for full workflow

---

**This Markdown is structured for reference alongside your repo, VS Code project, and GitHub WORKFLOW.md, supporting you through each daily task of the 90-day roadmap.**

---

<!-- Reference Links -->
[1]: https://learn.microsoft.com/en-us/windows/wsl/install
[2]: https://code.visualstudio.com/docs/remote/wsl
[3]: https://docs.github.com/en/get-started/quickstart/hello-world
[4]: https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
[5]: https://pypi.org/project/yfinance/
[6]: https://www.investopedia.com/terms/o/open-high-low-close-volume-ohlcv.asp
[7]: https://www.kaggle.com/datasets
[8]: https://www.freecodecamp.org/news/git-and-github-for-beginners/
[9]: https://pre-commit.com/
[10]: https://scikit-learn.org/stable/user_guide.html
[11]: https://www.kdnuggets.com/2020/03/7-ml-evaluation-metrics.html
[12]: https://github.com/microsoft/phi-3
[13]: https://huggingface.co/microsoft/phi-3-mini-128k-instruct
[14]: https://huggingface.co/docs/peft/index
[15]: https://arxiv.org/abs/2305.14314
[16]: https://www.deeplearning.ai/courses/retrieval-augmented-generation/
[17]: https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa
[18]: https://docs.pinecone.io/docs
[19]: https://github.com/facebookresearch/faiss
[20]: https://docs.datature.io/docs/pruning
[21]: https://github.com/pytorch/ao/blob/main/examples/pruning/pruning_example.ipynb
[22]: https://github.com/NervanaSystems/distiller
[23]: https://github.com/IST-DASLab/sparsegpt
[24]: https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction
[25]: https://python.langchain.com/docs/modules/prompts/strategies/
[26]: https://vsupalov.com/docker-python-poetry/
[27]: https://docs.github.com/en/actions/quickstart
[28]: https://fastapi.tiangolo.com/tutorial/
[29]: https://grafana.com/docs/grafana/latest/getting-started/getting-started-grafana/

