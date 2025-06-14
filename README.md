# 🤖 Y‑DATA Data Analyst Agent

This repository contains an interactive data analysis agent built for the **Agentic Systems** course, part of the **Y‑DATA 2024–2025** program.

It’s a simple Streamlit application powered by an agent that can answer user questions about the [Bitext – Customer Service Tagged Training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

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
# Edit .env and insert your NEBIUS_STUDIO_API_KEY
```

### 4. Run the app
```bash
streamlit run DataAnalyst.py
```
