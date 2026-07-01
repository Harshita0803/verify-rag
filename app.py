"""
Verify-RAG: A Human-in-the-Loop Framework for Citation and Groundedness
Verification of Retrieval-Augmented Generation.

Live demo implementing the three intervention points (IP-1 retrieval,
IP-2 groundedness, IP-3 citation) from the paper, plus a results page
reporting the real RAGTruth benchmark evaluation.

Author: Harshita Sharma
License: MIT
"""

import gradio as gr

# ----------------------------------------------------------------------
# Automated detection layer: LettuceDetect (loaded lazily so the Space
# starts fast and the Results tab works even before the model downloads).
# ----------------------------------------------------------------------
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from lettucedetect.models.inference import HallucinationDetector
        _detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1",  # base = smaller/faster on free CPU
        )
    return _detector


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def split_into_claims(answer):
    """Naive sentence segmentation into claim units for IP-2."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


def run_verification(sources_text, question, answer):
    if not sources_text.strip() or not answer.strip():
        return (
            "Please paste both the retrieved sources and the generated answer.",
            "", "",
        )

    passages = [p.strip() for p in sources_text.split("\n\n") if p.strip()]
    if not passages:
        passages = [sources_text.strip()]

    # ---- IP-1: Retrieval Validation -------------------------------------
    ip1 = (
        f"**IP-1 · Retrieval Validation**\n\n"
        f"{len(passages)} passage(s) retrieved. A human reviewer confirms these are "
        f"relevant and sufficient to answer the question before groundedness is judged.\n\n"
    )
    for i, p in enumerate(passages, 1):
        preview = p if len(p) < 240 else p[:240] + " …"
        ip1 += f"> **Passage {i}:** {preview}\n\n"

    # ---- IP-2: Claim-level Groundedness (LettuceDetect) -----------------
    try:
        detector = get_detector()
        spans = detector.predict(
            context=passages,
            question=question or "",
            answer=answer,
            output_format="spans",
        )
    except Exception as e:
        spans = []
        ip2_err = f"\n\n_(Detector unavailable: {e})_"
    else:
        ip2_err = ""

    flagged = [s for s in spans if isinstance(s, dict)]
    ip2 = "**IP-2 · Claim-Level Groundedness Check**\n\n"
    if flagged:
        ip2 += (
            f"The automated detector flagged **{len(flagged)}** span(s) as potentially "
            f"unsupported by the passages. A reviewer verifies each before the answer is trusted:\n\n"
        )
        for s in flagged:
            txt = s.get("text", "")
            conf = s.get("confidence", None)
            conf_str = f" _(confidence {conf:.2f})_" if isinstance(conf, (int, float)) else ""
            ip2 += f"- ⚠️ “{txt}”{conf_str}\n"
        ip2 += (
            "\n> **Reviewer note:** roughly 1 in 5 automated flags is a false alarm "
            "(precision ≈ 0.81 on RAGTruth), so human confirmation here removes false positives."
        )
    else:
        ip2 += (
            "No unsupported spans were flagged. **Reviewer note:** the detector misses ~22% of "
            "hallucinations (recall ≈ 0.78), so a reviewer should still skim for subtle claims "
            "the model did not catch."
        )
    ip2 += ip2_err

    # ---- IP-3: Citation Alignment ---------------------------------------
    import re
    cited = sorted(set(re.findall(r"\[(\d+)\]", answer)))
    ip3 = "**IP-3 · Citation Alignment Verification**\n\n"
    if cited:
        ip3 += (
            f"The answer cites source(s): {', '.join('[' + c + ']' for c in cited)}. "
            f"A reviewer confirms each cited passage actually supports the claim it is attached to "
            f"— models frequently cite sources that do not substantiate the statement.\n\n"
        )
        for c in cited:
            idx = int(c)
            if 1 <= idx <= len(passages):
                ip3 += f"- [{c}] → maps to Passage {idx} ✔ (reviewer verifies support)\n"
            else:
                ip3 += f"- [{c}] → ⚠️ no matching passage found (possible dangling citation)\n"
    else:
        ip3 += (
            "No inline citations (e.g. `[1]`) detected in the answer. If your RAG system attaches "
            "citations, include them as `[n]` markers to enable this check."
        )

    return ip1, ip2, ip3


# ----------------------------------------------------------------------
# Results / benchmark page (real numbers from the paper)
# ----------------------------------------------------------------------
RESULTS_MD = """
## Benchmark results (RAGTruth test set)

These are the real evaluation results reported in the paper. Every number is
reproducible from the released evaluation notebook.

### LettuceDetect (large) — example-level detection

| Task | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Question answering | 900 | 0.659 | 0.750 | 0.702 |
| Summarization | 900 | 0.669 | 0.554 | 0.606 |
| Data-to-text | 900 | 0.892 | 0.874 | 0.883 |
| **Overall (pooled)** | **2,700** | **0.805** | **0.784** | **0.794** |

The pooled F1 of **0.794** matches LettuceDetect's published benchmark performance,
confirming the evaluation harness is correct.

### Specialized detector vs. prompt-based judge (QA subset)

| Method | n | Precision | Recall | F1 |
|---|---|---|---|---|
| LettuceDetect (large) | 900 | 0.659 | 0.750 | 0.702 |
| GPT-4o-mini prompt judge | 200 | 0.375 | 0.545 | 0.444 |

A fine-tuned span detector substantially outperforms a naive single-call LLM judge.

### Why a human is still needed

Even the stronger detector leaves **~20% false alarms** and misses **~22% of
hallucinations**. In accountability-sensitive settings that residual error is why
Verify-RAG places a structured human check at the three intervention points above —
rather than trusting automation alone.
"""

INTRO_MD = """
# Verify-RAG

**A Human-in-the-Loop Framework for Citation and Groundedness Verification of
Retrieval-Augmented Generation.**

RAG grounds LLM answers in retrieved sources, but hallucinations persist and even the
best automated detectors miss ~1 in 5. Verify-RAG structures human verification at
three points: **IP-1** retrieval validation, **IP-2** claim-level groundedness
checking (automated flagging + human confirmation), and **IP-3** citation alignment.

Paste a RAG answer and its sources in the **Verifier** tab to see the workflow, or
open the **Results** tab for the RAGTruth benchmark.

*Author: Harshita Sharma · Paper + code linked in the repository.*
"""

EXAMPLE_SOURCES = (
    "Passage 1: The Eiffel Tower was completed in 1889 for the World's Fair in Paris. "
    "It stands 330 metres tall including antennas.\n\n"
    "Passage 2: The tower was designed by the engineering firm of Gustave Eiffel."
)
EXAMPLE_ANSWER = (
    "The Eiffel Tower was completed in 1889 [1] and is 330 metres tall [1]. "
    "It was designed by Gustave Eiffel's firm [2] and is made entirely of gold."
)

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
with gr.Blocks(title="Verify-RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MD)

    with gr.Tabs():
        with gr.Tab("Verifier"):
            gr.Markdown(
                "Paste the **retrieved sources** (separate passages with a blank line) and the "
                "**generated answer**. Use `[1]`, `[2]` … in the answer to mark citations."
            )
            with gr.Row():
                with gr.Column():
                    sources = gr.Textbox(
                        label="Retrieved sources / passages",
                        lines=8, value=EXAMPLE_SOURCES,
                    )
                    question = gr.Textbox(label="Question (optional)", lines=1)
                    answer = gr.Textbox(
                        label="Generated answer", lines=5, value=EXAMPLE_ANSWER
                    )
                    btn = gr.Button("Run three-layer verification", variant="primary")
                with gr.Column():
                    out1 = gr.Markdown()
                    out2 = gr.Markdown()
                    out3 = gr.Markdown()
            btn.click(run_verification, [sources, question, answer], [out1, out2, out3])

        with gr.Tab("Results"):
            gr.Markdown(RESULTS_MD)

if __name__ == "__main__":
    demo.launch()
