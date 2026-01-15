# 🚀 BenchPress

A prototype system for transforming real-world enterprise SQL logs into high-quality Text-to-SQL benchmarks using human-in-the-loop annotation workflows.

<p align="center">
  <img src="demo/preview.png" width="600" alt="Demo Screenshot">
</p>

---

## 🔍 Overview

**Enterprise Text-to-SQL** addresses the challenge of building realistic, domain-specific Text-to-SQL datasets by combining:
- SQL log mining
- Human-in-the-loop annotation
- LLM-assisted generation and validation

This system was developed as part of the BENCHPRESS project and supports benchmark creation from internal SQL query logs.

---

## 📺 Demo & Deployment

- **Live Demo:** _Coming soon_ <!-- [Coming soon] or [deployment link if hosted] -->
- **Video Walkthrough:** [▶ Watch on YouTube](https://www.youtube.com/coming soon)
- **Poster Presentation:** [View the _NEDB 2025 Poster_(PDF)](demo/BENCHPRESS-NEDB-poster.pdf)

---

## 📦 Features

- ✅ Upload and parse enterprise SQL logs  
- ✅ Auto-cluster similar queries using LLM embeddings  
- ✅ Generate natural language annotations with prompt-based LLMs  
- ✅ Verify and edit annotations via an easy-to-use UI  
- ✅ Export clean Text-to-SQL benchmark datasets  

---

## 🔧 Installation

```bash
git clone https://github.com/fabian-wenz/enterprise-txt2sql.git
cd enterprise-txt2sql
pip install -r requirements.txt
```

Requirements:
- Python 3.9+
- OpenAI API Key (or compatible LLM provider)
- Optional: [pgvector](https://github.com/pgvector/pgvector) for clustering via vector search

---

## 🚀 Quickstart

```bash
python website/app.py
```


Then open your browser and go to:  
[http://localhost:8000](http://localhost:8000)

---

## 🧠 Annotation Workflow

1. **Project Setup**: Create a new annotation project for a specific enterprise workload.
2. **Data Ingestion**: Upload SQL logs and schema files, or select a public benchmark (Bird, FIBEN, Spider, Beaver).
3. **Task Configuration**: Select annotation direction (SQL→NL) and a language model (e.g., GPT-4o, GPT-3.5, DeepSeek).
4. **(Optional) Decomposition**: Split nested SQL into simpler subqueries using CTEs.
4. **Context Retrieval**: Retrieve similar annotated examples and relevant tables using dense embeddings.
5. **Candidate Generation**: LLM generates 4 NL candidates using retrieval-augmented few-shot prompting.
6. **(Optional) Recomposition**: Merge subquery descriptions into a single coherent explanation.
6. **Human Feedback**: Annotators rank, edit, or discard LLM outputs.
7. **Review & Export**: Export final annotations for training or evaluation; optionally auto-evaluate if ground truth exists.

<p align="center">
  <img src="demo/workflow.jpeg" width="700" alt="Annotation Workflow">
</p>

---

## 📁 Project Structure

```text
.
├── demo/                # Screenshots and videos for README
├── website/             # Preprocessing, clustering, and evaluation scripts
  ├── data/                # Sample SQL logs and generated benchmark data
  ├── templates/           # HTML templates for visualizinf the website
  ├── app.py               # Main entry point for the UI
  ├── config.py            # Prompts and LLM interaction
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📊 Example Output

```json
{
  "question": "Show the top 10 customers by revenue.",
  "query": "SELECT customer_name FROM sales ORDER BY revenue DESC LIMIT 10"
}
```

---

## 📜 Paper

> **BENCHPRESS: An Annotation System for Rapid Text-to-SQL Benchmark Curation**  
> Fabian Wenz*, Peter Baile Chen, Moe Kayali, Michael Stonebraker, Cagatay Demiralp  
> _Submitted to CIDR 2026_  
> [📄 coming soon on](https://)

---

## 🙌 Acknowledgements

This project was developed during Fabian Wenz’s time at MIT CSAIL with the support of:

- Prof. Michael Stonebraker  
- Dr. Cagatay Demiralp  
- Peter Baile Chen  
- Dr. Nesime Tatbul

---

## 🛠️ Contributing

We welcome contributions from the community!

If you encounter bugs, want to request features, or contribute code, please:
- Submit an issue
- Fork the repo and open a pull request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more details.