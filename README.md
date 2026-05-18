# Entity Resolver (LLM Entity Resolution Pipeline)

This project builds a production-minded entity resolution pipeline for CRM deduplication and cross-file matching. It combines fast embedding-based **blocking** (to reduce the search space from N² to N·K) with an LLM **verification** step that makes high-precision match decisions using the full record context—typically outperforming brittle rule-based and heuristic-only approaches.

## Setup

```bash
cd entity-resolver
pip install -r requirements.txt
```

Create a `.env` file.

OpenAI API (default):

```bash
cp .env.example .env
# edit .env
```

Local Ollama (Gemma 4 / Gemma 3):

- Install Ollama and pull a model (example): `ollama pull gemma4:e2b` or `ollama pull gemma3:4b`
- Set:
  - `LLM_PROVIDER=ollama`
  - `LLM_CHAT_COMPLETIONS_URL=http://localhost:11434/v1/chat/completions`
  - `LLM_MODEL=gemma4:e2b` (or `gemma3:4b` for smaller VRAM)
- If your PC freezes or thrashes (common when too many requests hit Ollama at once), set conservative limits:
  - `LLM_CONCURRENCY=1`
  - `LLM_BATCH_SIZE=2`
  - `LLM_MAX_TOKENS=200`
  - Optional: `OLLAMA_NUM_CTX=2048` (lower context reduces VRAM/RAM pressure)

## Benchmarks

Download benchmark datasets (Ditto S3 mirrors of ER-Magellan):

```bash
python benchmarks/download.py
```

Run the baseline comparison (writes `benchmarks/results.txt`):

```bash
python benchmarks/run_baseline.py
```

Notes:
- `dedupe.io` training is mocked by default (interactive labeling would otherwise block automation).
- If `LLM_PROVIDER=openai`, the pipeline requires `OPENAI_API_KEY` (or `LLM_API_KEY`) to run the LLM verification stage; without it, the benchmark script will still generate a table with placeholders.

### F1 gap (placeholder)

| Dataset        | Method       | Precision | Recall | F1  |
|---------------|--------------|-----------|--------|-----|
| DBLP-ACM       | dedupe.io    |   0.xx    |  0.xx  |0.xx |
| DBLP-ACM       | Our pipeline |   0.xx    |  0.xx  |0.xx |
| Amazon-Google  | dedupe.io    |   0.xx    |  0.xx  |0.xx |
| Amazon-Google  | Our pipeline |   0.xx    |  0.xx  |0.xx |

## Run on your CSV

Dedup a single file:

```bash
python intake.py --file contacts.csv --mode dedup --threshold 0.85 --output report.html
```

Match two files:

```bash
python intake.py --file_a crm.csv --file_b import.csv --mode match --threshold 0.85 --output report.html
```

If your CSV does not contain an `id` column, row index will be used automatically.
