# 🤖 Y‑DATA Data Analyst Agent

This repository contains an interactive data analysis agent built for the **Agentic Systems** course, part of the **Y‑DATA 2024–2025** program.

A lightweight Streamlit application powered by an LLM-based agent that can answer user questions about the [Bitext – Customer Service Tagged Training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), which contains labeled support dialogues for training chatbot systems.

### 🧠 Supported Question Types

The agent handles three types of questions:

1. **Structured**  
   Examples:
   - What are the most frequent categories?
   - Show examples of Category X
   - What categories exist?
   - Show intent distributions

2. **Unstructured**  
   Examples:
   - Summarize Category X
   - Summarize how agents respond to Intent Y

3. **Out-of-scope**  
   Examples:
   - Who is Magnus Carlson?
   - What is Serj's rating?

---

## ⚙️ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/OphirTuretz/Y-DATA-Data-Analyst-Agent.git
cd Y-DATA-Data-Analyst-Agent
```

### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env_template .env
# Edit .env and insert your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 4. Run the app
```bash
streamlit run DataAnalyst.py
```

---

## 🧱 Architecture Diagram
<p align="center">
  <img src="images/Architecture_Diagram.svg" alt="System architecture diagram showing components and flow" />
</p>
