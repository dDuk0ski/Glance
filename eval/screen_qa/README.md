# Screen-QA Benchmark

Measures how well different models answer questions about what's on screen, which is what Glance's eye toggle does. The results pick the app's default model and screenshot size, and form the main model-evaluation section of the report.

This folder is platform-independent Python. It calls models through OpenRouter, the same API the app uses.

## Setup

```bash
cd eval/screen_qa
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your OpenRouter key in .env
```

Never commit `.env`. Set a spending limit on the OpenRouter key.

The OCR condition needs the Tesseract binary: `brew install tesseract` (macOS), `sudo apt install tesseract-ocr` (Linux), or the Windows installer from the Tesseract project.

## Layout

```
eval/screen_qa/
  data/
    images/         # screenshots (committed; see privacy note)
    items.jsonl     # one question per line
  run.py            # runs the experiment grid, writes raw results
  score.py          # exact match + LLM-judge scoring
  analyze.py        # tables and plots
  results/          # raw outputs, scores, plots for the report
```

## Dataset

50–100 desktop screenshots with questions. Cover a mix of apps (browser, IDE, terminal, docs, spreadsheets, settings, chats) and question types:

| category | example |
|---|---|
| `read` | "What's the total in cell D12?" |
| `locate` | "Which menu has the export option?" |
| `explain` | "What does this error mean?" |
| `summarize` | "Summarize this email thread." |
| `code` | "Why does this function return None?" |

`data/items.jsonl`, one object per line:

```json
{"id": "q001", "image": "images/vscode_error_01.png", "question": "What line is the error on?", "answer": "42", "answer_type": "exact", "category": "read", "app": "vscode"}
```

`answer_type` is `exact` (short factual answer, string-matched after normalizing) or `open` (free-form, graded by the LLM judge against `answer`).

**Privacy:** this repo is public. Make screenshots on purpose for the dataset: no real names, emails, messages, API keys or account info.

## Experiment grid

For each item, run every combination of:

- **model:** 4–5 vision-capable models on OpenRouter, across different providers and price points
- **image size:** longest side 512, 1024, 1568 px, and full resolution
- **input:** `image` (screenshot only), `ocr` (Tesseract text only), `both`

Use the same system prompt for every run, and log everything raw so scoring can be re-run without new API calls.

`results/raw.jsonl`, one line per run:

```json
{"id": "q001", "model": "...", "size": 1024, "input": "image", "response": "...", "ttft_s": 0.8, "total_s": 2.4, "prompt_tokens": 1100, "completion_tokens": 60, "cost_usd": 0.0004}
```

Use streaming so time to first token (`ttft_s`) can be measured; that's what the user feels in the notch. Compute cost from token usage and OpenRouter's listed prices.

## Scoring

- **`exact` items:** normalized string match
- **`open` items:** LLM judge (a model not under test) rates each answer correct / partially correct / incorrect against the reference
- **Validate the judge:** hand-grade about 20% of the open answers and report how often you agree with the judge

## What to report

- Accuracy per model, image size, input type and category
- Latency (time to first token and total) and cost per query
- Accuracy vs cost and accuracy vs latency charts, to show the trade-off
- Where image beats OCR and vice versa (e.g. dense text vs charts and UI layout)

Note: the app would use Apple Vision for OCR, not Tesseract. Mention this in the report, or run Vision OCR text through the same grid as an extra condition.

## Deliverables

- Recommended default model and screenshot size for the app
- Tables and plots in `results/`
- The screen-QA sections of the report (Data, Experiments & Results)
