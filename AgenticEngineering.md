# 90-Day Plan: Code Editor Plugin for Agentic Workflow with Python & Small Models

**Objective:**  
Develop a VS Code plugin that orchestrates AI agent workflows using Python and a small LLM, mastering agentic development step by step while using open source tools, GitHub, and WSL.

---

## 📅 Phase Overview

| Phase           | Days  | Focus                                  | Example Deliverable                                |
|-----------------|-------|----------------------------------------|----------------------------------------------------|
| Foundations     | 1–14  | Python, VS Code plugin, WSL setup      | Hello-world plugin, agent Python module            |
| Agent Basics    | 15–28 | Agentic concepts, core agent in Python | Standalone agent: docstring/gen/search in files    |
| Agent+Plugin    | 29–42 | Plugin-agent integration, LLM access   | Agent runs inside VS Code, LLM small model support |
| Workflows       | 43–56 | Task chaining, memory, orchestration   | Editor workflow: explain → suggest → refactor      |
| Small Models    | 57–70 | Efficient local LLM, fallback          | Local/remote model switch, RAM-optimized runs      |
| Evaluation/UX   | 71–80 | UX prompts, testing                    | Prompt selector UI, test suite, usability audit    |
| Adv. Agentics   | 81–90 | Planning, extensibility, docs          | Modular agents, docs, backlog, feedback channel    |

---

## 🛠️ Before You Start: Prerequisites

- [Install WSL on Windows](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Set up VS Code with WSL](https://code.visualstudio.com/docs/remote/wsl)
- [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Clone/Init GitHub repo](https://docs.github.com/en/get-started/quickstart/hello-world)
- [VS Code Extension Scaffolding Guide](https://www.kdnuggets.com/build-your-first-python-extension-for-vs-code-in-7-easy-steps)[8], [YouTube Example Walkthrough](https://www.youtube.com/watch?v=b9iUyzmgQUY)[14]

---

## 🔗 Core Resources and References

| Topic           | Resource                                                                                      |
|-----------------|----------------------------------------------------------------------------------------------|
| Open Source Agent Extension | [Kilo Code Plugin](https://kilocode.ai)[1], [Blinky Agent Debugging](https://github.com/e2b-dev/awesome-ai-agents)[5]                 |
| Agentic Workflow Tutorials  | [LangGraph & LangChain AI Agents Guide](https://www.codecademy.com/article/agentic-ai-with-langchain-langgraph)[6], [TalkPython FM LangGraph Interview](https://talkpython.fm/episodes/show/507/agentic-ai-workflows-with-langgraph)[12] |
| CrewAI            | [CrewAI Framework: Concepts](https://arize.com/docs/ax/learn/guides/readme/crewai)[7][13], [Docs](https://docs.crewai.com)[19]       |
| Small LLMs Python | [Hands-on SLM Python Guide](https://ai.gopubby.com/a-hands-on-guide-to-training-your-own-small-language-model-slm-in-python-c189776f87c2?gi=11c38bd1ff4f)[9], [SLM Video & Source](https://xbe.at/index.php?filename=Introduction%20to%20Small%20Language%20Models%20%28SLMs%29%20in%20Python.md)[15] |
| Best Agent Practices  | [How to Build AI Agents That Work](https://www.geeky-gadgets.com/building-ai-agents-best-practices/)[10]           |
| VS Code AI Agent Examples | [Open Source Editor Milestone](https://code.visualstudio.com/blogs/2025/06/30/openSourceAIEditorFirstMilestone)[2], [Free Local AI Assistant YouTube](https://www.youtube.com/watch?v=he0_W5iCv-I)[3]     |

---

## 🗓️ Detailed 90-Day Breakdown

### **Days 1–14: Environment & Plugin Foundations**
- Install VS Code, WSL, Python, and Node.js.
- Scaffold a hello-world VS Code extension:  
  - Use [`yo code`](https://www.kdnuggets.com/build-your-first-python-extension-for-vs-code-in-7-easy-steps)[8] to generate template.
- Create `core_agent/` with standalone Python agent (does docstring/gen/search).
- Track everything in git and push daily to GitHub.
- Reference: [VS Code Python Getting Started][20], [Extension Scaffold[8]][8], [Plugin Walkthrough][14].

### **Days 15–28: Agent Essentials in Python**
- Learn [agentic pattern fundamentals](https://ai.gopubby.com/a-hands-on-guide-to-training-your-own-small-language-model-slm-in-python-c189776f87c2?gi=11c38bd1ff4f)[9]:  
  - Roles, tasks, tool calls, progress plans.
- Implement agent as a Python class:  
  - Respond to text commands, edit code, summarize, search.
- Refactor for reusability, add dependency management.
- Reference: [SLM Beginner’s Guide Video][15], [Agentic Best Practices][10].

### **Days 29–42: Plugin + Agent Integration with LLMs**
- Connect your agent to the plugin—let users run commands from VS Code sidebar or right-click menu.
- Integrate a small LLM locally with [Hugging Face’s Transformers](https://huggingface.co/transformers/) or run [LLMs in Python](https://xbe.at/index.php?filename=Introduction%20to%20Small%20Language%20Models%20%28SLMs%29%20in%20Python.md)[15].
- Add toggles to switch between local and API LLM fallback.
- Reference: [Open Source Plugin Examples][1][2][3], [Language Model Python Integration][9][15].

### **Days 43–56: Orchestrating Agentic Workflows**
- Build multi-step workflows using [LangGraph](https://www.codecademy.com/article/agentic-ai-with-langchain-langgraph)[6] or [CrewAI’s sequential, hierarchical strategies](https://arize.com/docs/ax/learn/guides/readme/crewai)[7][13].
- Sample workflow: explain code → suggest change → refactor → test.
- Design logs/trace panel for action visibility in UI.
- Reference: [LangGraph/CrewAI Docs][7][12][13][19], [Hands-On Guide][6].

### **Days 57–70: Optimize for Small Models**
- Experiment with more compressed LLMs (DistilBERT, TinyBERT, MiniLM).
- Profile RAM/latency trade-offs, and cache results for efficiency.
- Let users choose between different model engines in plugin settings.
- Reference: [TinyBERT Paper](https://arxiv.org/abs/2002.11269), [Python SLM Starter][9][15].

### **Days 71–80: UX, Prompting & Testing**
- Build prompt engineering UI: let user select/update agent prompts/templates.
- Write test coverage for edge cases and plugin errors.
- Add UI for submitting agent feedback or usability bugs.
- Reference: [Prompt Optimization in Agents][10], [LangChain Advanced Patterns](https://python.langchain.com/docs/modules/prompts/strategies/).

### **Days 81–90: Advanced Context, Planning & Docs**
- Modularize agent logic for easy extension (commands: refactor, lint, docify, test).
- Add planning: agent crafts dependency/work subschedules before acting ([CrewAI Planning][7][13][19]).
- Write inline tutorial, usage docs, GitHub issues for user feedback.
- Release plugin beta for peer review.
- Reference: [CrewAI Docs][19], [Advanced LangGraph][6][12][18].

---

## 📖 Must-Reads for Each Topic

- **Agentic Concepts:**  
  - [How to Build AI Agents – Best Practices][10]
  - [AI Agentic Workflows with LangGraph][12]

- **Hands-On Plugins & Agents:**  
  - [Kilo Code – AI Agent VS Code Plugin][1]
  - [GitHub Copilot Chat, open-source][2]
  - [Blinky Debugger for VS Code][5]

- **Agent Frameworks & Multi-Agent:**  
  - [Agentic AI with LangGraph][6]
  - [CrewAI Multi-Agent Orchestration][7][13][19]

- **Python + Small LLMs:**  
  - [Hands-on SLM in Python][9]
  - [TinyBERT in Python][15]

- **Extending & Integrating in VS Code:**  
  - [Step-by-step Python Extension Guide][8][14]
  - [VS Code Python Docs][20]

---

## 📝 Example Milestone Checkpoints

| Day  | Progress Example                                     |
|------|------------------------------------------------------|
| 14   | Minimal plugin triggers standalone Python agent       |
| 28   | Agent runs explain/gen/search on file in editor       |
| 42   | Plugin offers LLM-backed summaries, code suggestions  |
| 56   | Multi-step workflow ("explain → refactor → test")     |
| 70   | Efficient runs on local MiniLM/DistilBERT, cacheable  |
| 80   | Usable UI for prompts, context, and edge handling     |
| 90   | Modular, documented, testable beta plugin on GitHub   |

---

## 🏁 Pro Tips

- Use the [Kilo Code open source plugin][1] to study full-featured orchestration patterns and LLM tool integration.
- [CrewAI][7][13][19] and [LangGraph][6][12][18] enable high-level workflow graphs and multi-agent tasking—excellent for learning advanced agentic patterns.
- Write your plugin UI in small, testable increments—incrementally add agent skills/modules.
- Log agent actions and user feedback for real-world learning.
- Use [Blinky][5] as a reference for agent debugging patterns.

---

## 🎒 Further Learning & Challenges

- Add support for new languages/tools (see [Awesome AI Agents][5]).
- Explore “Human-in-the-Loop” for agent approval phases ([LangGraph AI Workflows][12]).
- Experiment with external tool calls from your agent (browsing, code search).
- Add multi-agent "Crew" model for collaborative code tasks ([CrewAI][19]).

---
