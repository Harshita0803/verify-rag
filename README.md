---
title: Verify-RAG
emoji: 🔎
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
python_version: 3.11
app_file: app.py
pinned: false
license: mit

---

# Verify-RAG

**A Human-in-the-Loop Framework for Citation and Groundedness Verification of Retrieval-Augmented Generation.**

Retrieval-Augmented Generation (RAG) grounds LLM outputs in retrieved sources, yet hallucinations persist — and even the strongest automated detectors miss roughly one in five. Verify-RAG structures human verification of RAG outputs at **three intervention points**:

- **IP-1 — Retrieval Validation:** are the retrieved passages relevant and sufficient?
- **IP-2 — Claim-Level Groundedness Check:** are the answer's claims supported by the passages? (automated flagging with [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) + human confirmation)
- **IP-3 — Citation Alignment Verification:** do cited sources actually support the claims they are attached to?

This repository contains a working implementation (a Gradio app) and the evaluation used in the accompanying paper.

## Live demo

Paste a RAG answer and its sources into the **Verifier** tab to walk through the three-layer workflow, or open the **Results** tab for the benchmark.

## Benchmark results (RAGTruth test set)

Example-level hallucination detection with LettuceDetect (large). Every number is reproducible from the released evaluation notebook.

| Task | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Question answering | 900 | 0.659 | 0.750 | 0.702 |
| Summarization | 900 | 0.669 | 0.554 | 0.606 |
| Data-to-text | 900 | 0.892 | 0.874 | 0.883 |
| **Overall (pooled)** | **2,700** | **0.805** | **0.784** | **0.794** |

The pooled F1 of **0.794** matches LettuceDetect's published benchmark performance, confirming the evaluation harness. On the QA subset, the fine-tuned span detector (F1 0.702) substantially outperforms a prompt-based GPT-4o-mini judge (F1 0.444).

## Scope and honesty note

This is a **framework and working implementation** plus a **reproduction-grade benchmark evaluation**. The automated detection layer is implemented and evaluated. The human-study component (reviewer effort, trust calibration) is specified as a **pre-registered protocol** in the paper and has **not yet been run** — those results are future work, not claimed here.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## Citation

If you use this framework, please cite the accompanying paper: https://doi.org/10.5281/zenodo.21118817

## License

MIT © Harshita Sharma
