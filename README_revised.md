# Jev-Compatible Typed Evaluation Service

![Python](https://img.shields.io/badge/Python-3.12.7-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Tests](https://img.shields.io/badge/offline%20tests-179%20passed-brightgreen)
![Benchmark](https://img.shields.io/badge/benchmark-28%20examples-blue)

A small, stateless HTTP service that reproduces the core shape of Jev's typed evaluation API with a general-purpose GPT model, then evaluates the local implementation and Jev on the same frozen labeled dataset.

The repository focuses on three things:

- a clean typed evaluation API,
- reliable model-output validation and normalization,
- a reproducible side-by-side benchmark against Jev.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Question Types](#question-types)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Reliability and Error Handling](#reliability-and-error-handling)
- [Configuration](#configuration)
- [Evaluation Methodology](#evaluation-methodology)
- [Benchmark Results](#benchmark-results)
- [Error Analysis](#error-analysis)
- [Reproducing the Benchmark](#reproducing-the-benchmark)
- [Testing](#testing)
- [Result Artifacts](#result-artifacts)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Dataset Attribution](#dataset-attribution)
- [License](#license)

---

## What This Project Does

The service exposes a Jev-style endpoint:

```text
POST /v1/systemone
```

A request contains:

- one shared `state`,
- a local model alias,
- one or more named evaluation questions.

Each question is evaluated independently and returned as a typed answer.

The current implementation supports:

- **Choice** — select one option from named criteria and return normalized probabilities plus confidence.
- **Noul** — return a yes/no likelihood, a derived boolean answer, and confidence.

The same labeled dataset is then sent through both:

1. **Jev**
2. **the local service**

Both runners write a shared canonical result format, and a separate offline comparison step computes accuracy, coverage, boolean metrics, confidence summaries, and error analysis.

This keeps model execution separate from scoring and makes the final comparison reproducible.

---

## Highlights

- **Jev-style request model** with shared `state` and multiple named questions.
- **Typed Choice and Noul responses**.
- **FastAPI + Pydantic v2** request and response validation.
- **OpenAI Structured Outputs** for constrained provider responses.
- **Semantic normalization** beyond schema validation.
- **Service-owned retry policy** with SDK automatic retries disabled.
- **Safe public error responses** without leaking raw provider details.
- **Stateless design** — no database, queue, session store, or background worker required.
- **Ground-truth isolation** — expected labels and source metadata are never sent to the evaluated model/service.
- **Shared benchmark infrastructure** for Jev and Local runs.
- **Hash-checked evaluation inputs** for reproducibility.
- **Failure-preserving metrics** — failed calls are retained rather than silently dropped.

---

# Quick Start

## 1. Requirements

Validated environment:

```text
Python 3.12.7
```

You also need:

- an OpenAI API key for the local GPT-backed evaluator,
- a Jev API key only if you want to reproduce the Jev benchmark,
- network access for live provider calls.

No database or Docker setup is required.

## 2. Create an environment

Using `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -m pip check
```

## 4. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

At minimum, a local live run needs:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

A Jev benchmark run additionally needs:

```dotenv
JEV_API_KEY=
JEV_MODEL=jev-latest
```

Do not commit `.env`.

## 5. Start the local service

```bash
python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Interactive FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI document is available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

# API Usage

## Endpoint

```text
POST /v1/systemone
```

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/v1/systemone" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The deployment passed validation and is ready to release.",
    "model": "gpt-local",
    "questions": {
      "release_decision": {
        "type": "choice",
        "instruction": "What should happen next?",
        "criteria": {
          "approve": "The release is ready to proceed.",
          "revise": "The release still needs additional work."
        }
      },
      "has_confirmation": {
        "type": "noul",
        "instruction": "Does the state explicitly confirm successful validation?"
      }
    }
  }'
```

## Example response shape

```json
{
  "model": "gpt-4o-mini",
  "answers": {
    "release_decision": {
      "type": "choice",
      "choice": "approve",
      "probabilities": {
        "approve": 0.95,
        "revise": 0.05
      },
      "confidence": 0.95
    },
    "has_confirmation": {
      "type": "noul",
      "noul": 0.9,
      "answer": true,
      "confidence": 0.9
    }
  }
}
```

The exact model output can vary. The important part is the typed response contract.

---

# Question Types

## Choice

A Choice question contains:

- `type: "choice"`
- an instruction
- a mapping of option names to descriptions

Example:

```json
{
  "type": "choice",
  "instruction": "What should happen next?",
  "criteria": {
    "approve": "Proceed with the release.",
    "revise": "Make additional changes first."
  }
}
```

The normalized answer contains:

```json
{
  "type": "choice",
  "choice": "approve",
  "probabilities": {
    "approve": 0.95,
    "revise": 0.05
  },
  "confidence": 0.95
}
```

Validation includes:

- exact probability keys,
- finite probabilities,
- values in `[0, 1]`,
- nonzero probability mass,
- probability-sum tolerance,
- selected choice must be one of the maximum-probability options.

The local confidence is the maximum normalized probability.

## Noul

A Noul question contains:

- `type: "noul"`
- an instruction
- optional `true` / `false` criteria

Example:

```json
{
  "type": "noul",
  "instruction": "Does the state explicitly mention a location?"
}
```

The normalized local answer contains:

```json
{
  "type": "noul",
  "noul": 0.9,
  "answer": true,
  "confidence": 0.9
}
```

The boolean threshold is configurable and defaults to:

```text
0.5
```

For a probability `p`:

```text
answer = p >= 0.5
confidence = max(p, 1 - p)
```

---

# Architecture

The service is intentionally small and split into four layers.

```text
Client
  |
  | POST /v1/systemone
  v
+---------------------------+
| Entry Layer               |
| app/main.py               |
| HTTP + request/response   |
+-------------+-------------+
              |
              v
+---------------------------+
| Service Layer             |
| app/service.py            |
| orchestration + retry     |
+-------------+-------------+
              |
              v
+---------------------------+
| Model Layer               |
| app/model.py              |
| provider adapter          |
| Structured Outputs        |
+-------------+-------------+
              |
              v
+---------------------------+
| Normalization Layer       |
| app/normalization.py      |
| semantic validation       |
| typed local answers       |
+-------------+-------------+
              |
              v
        HTTP response
```

## 1. Entry Layer

`app/main.py`

Responsibilities:

- expose `POST /v1/systemone`,
- parse and validate HTTP requests,
- serialize successful responses,
- map internal service errors to safe public HTTP responses.

It does not contain provider logic.

## 2. Service Layer

`app/service.py`

Responsibilities:

- iterate over named questions,
- dispatch Choice vs Noul evaluation,
- own the retry budget,
- provide fresh state/question copies for retry attempts,
- call normalization,
- assemble the final answer map.

Retry ownership is centralized here so a request does not accidentally accumulate independent retry budgets across layers.

## 3. Model Layer

`app/model.py`

Responsibilities:

- define a provider-neutral evaluator interface,
- convert typed questions into provider requests,
- use OpenAI Structured Outputs,
- perform one provider attempt per call,
- map provider failures into controlled internal errors.

The OpenAI client is configured with:

```text
max_retries = 0
```

so retry policy remains under Service-layer control.

## 4. Normalization Layer

`app/normalization.py`

Responsibilities:

- semantic validation,
- probability normalization,
- Choice argmax validation,
- deterministic local confidence,
- Noul boolean conversion.

Normalization is pure:

- no network calls,
- no retries,
- no dataset-specific logic.

---

# Project Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── service.py              # orchestration and retry ownership
│   ├── model.py                # provider-neutral evaluator + OpenAI adapter
│   ├── normalization.py        # semantic validation and normalization
│   ├── schemas.py              # request/response and model-output schemas
│   ├── errors.py               # controlled service error hierarchy
│   └── config.py               # environment-backed settings
│
├── data/
│   ├── original/               # source SNIPS files
│   ├── evaluation_dataset.json # frozen 28-example benchmark
│   └── evaluation_tasks.json   # frozen Choice/Noul task definitions
│
├── docs/
│   ├── DECISIONS.md
│   ├── jev_compatibility.md
│   └── jev_service_architecture_plan.md
│
├── script/
│   ├── download_dataset.py
│   ├── build_dataset.py
│   ├── probe_jev.py
│   ├── evaluation_common.py
│   ├── run_jev.py
│   ├── run_local.py
│   └── compare_results.py
│
├── tests/
│   ├── fakes.py
│   ├── test_validation.py
│   ├── test_normalization.py
│   ├── test_model.py
│   ├── test_live_model.py
│   ├── test_service.py
│   ├── test_api.py
│   ├── test_evaluation_common.py
│   ├── test_run_jev.py
│   ├── test_run_local.py
│   ├── test_pipeline.py
│   └── test_metrics.py
│
├── results/
│   ├── jev_results.json
│   ├── local_results.json
│   └── metrics.json
│
├── .env.example
├── requirements.txt
└── README.md
```

The FastAPI application entrypoint is:

```text
app.main:app
```

---

# Reliability and Error Handling

The service uses several validation layers instead of trusting model output directly.

```text
Prompt constraints
      ↓
Structured Outputs
      ↓
Pydantic validation
      ↓
Semantic validation
      ↓
Typed normalized response
```

## Retry policy

Default:

```text
MAX_MODEL_RETRIES=1
```

That means a question receives at most:

```text
2 provider attempts
```

The same retry budget covers retryable model/provider failures rather than creating a new retry budget for each error category.

Retryable conditions include controlled transient/model-output failures.

Non-retryable conditions include cases such as configuration/authentication failures where repeating the same request would not help.

## Public error contract

Service errors are returned in a stable shape:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Safe public message"
  }
}
```

Representative status semantics:

| Status | Meaning |
| --- | --- |
| `422` | malformed or unsupported request |
| `502` | invalid/refused provider output or upstream configuration/auth failure |
| `503` | transient provider failure |
| `500` | unexpected internal error |

Raw API keys, full provider error bodies, and secrets are not included in public error messages.

---

# Configuration

Configuration is environment-backed. Project-root `.env` values are loaded when present, while process environment variables take precedence.

| Variable | Default / Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI credential for the local evaluator |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `JEV_API_KEY` | Jev credential for benchmark calls |
| `JEV_MODEL` | `jev-latest` |
| `JEV_BASE_URL` | `https://api.typesafe.ai` |
| `LOCAL_BASE_URL` | `http://127.0.0.1:8000` |
| `MAX_MODEL_RETRIES` | `1` |
| `MODEL_TIMEOUT_SECONDS` | `30` |
| `PROBABILITY_SUM_TOLERANCE` | `0.01` |
| `NOUL_BOOLEAN_THRESHOLD` | `0.5` |
| `MAX_QUESTIONS` | `10` |
| `LOG_LEVEL` | `INFO` |

Live scripts also require the explicit safety switch:

```text
RUN_LIVE_TESTS=1
```

This prevents accidental paid/network calls during normal offline testing.

---

# Evaluation Methodology

## Dataset

Source:

[`bkonkle/snips-joint-intent`](https://huggingface.co/datasets/bkonkle/snips-joint-intent)

The frozen evaluation set contains:

```text
28 examples
7 intents × 4 examples
```

The evaluated intent classes are:

- `AddToPlaylist`
- `BookRestaurant`
- `GetWeather`
- `PlayMusic`
- `RateBook`
- `SearchCreativeWork`
- `SearchScreeningEvent`

## Ground Truth

Two tasks are evaluated for every example.

### 1. Intent — Choice

Question:

> What is the user's intent?

Ground truth comes directly from the SNIPS intent label.

### 2. mentions_location — Noul

Question:

> Does the utterance explicitly mention a location?

The expected boolean is derived deterministically from location-related SNIPS slot annotations.

Jev output is never used as ground truth.

## Ground-truth isolation

Evaluation requests contain only:

- `state`
- `model`
- `questions`

They do **not** send:

- expected answers,
- source labels,
- source metadata,
- dataset IDs,
- test indices.

## Shared evaluation pipeline

```text
Frozen dataset + frozen task definitions
                 |
                 v
         shared request builder
          /               \
         v                 v
       Jev              Local API
         \                 /
          v               v
       canonical result records
                 |
                 v
         offline comparison
                 |
                 v
          results/metrics.json
```

Both runners share:

- dataset loading,
- task loading,
- request construction,
- expected-answer extraction,
- input hashing,
- canonical record format,
- atomic result writing.

Service-specific code is limited to transport/auth/model selection and native response parsing.

---

# Frozen Evaluation Inputs

Dataset SHA256:

```text
28c4f369a01637a1bf123a6930dfbccd9e6c0015cd31051fb8ed857ab891ea2d
```

Task-definition SHA256:

```text
299f0e3d8450a11debef72eba000c6d33e1183e35b870be83d58c02f36bb838b
```

Formal-run configuration:

```text
JEV_MODEL=jev-latest
OPENAI_MODEL=gpt-4o-mini
NOUL_BOOLEAN_THRESHOLD=0.5
PROBABILITY_SUM_TOLERANCE=0.01
MAX_MODEL_RETRIES=1
```

The comparison tool verifies input hashes before producing metrics.

---

# Benchmark Results

The completed frozen benchmark contained:

```text
28 examples
56 question-level predictions per service
```

Both services completed with:

```text
coverage = 1.0
failures = 0
```

## Task-level results

| Service | Task | Correct | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Jev | `intent` | 28/28 | 1.0000 | — | — | — |
| Jev | `mentions_location` | 28/28 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Local | `intent` | 27/28 | 0.9643 | — | — | — |
| Local | `mentions_location` | 25/28 | 0.8929 | 1.0000 | 0.7500 | 0.8571 |

## Overall

| Service | Correct | Total | Accuracy | Coverage | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev | 56 | 56 | 1.0000 | 1.0000 | 0 |
| Local | 52 | 56 | 0.9286 | 1.0000 | 0 |

## Pairwise outcomes

| Outcome | Count |
| --- | ---: |
| Both correct | 52 |
| Jev correct / Local incorrect | 4 |
| Local correct / Jev incorrect | 0 |
| Both incorrect | 0 |
| One or both failed | 0 |

These are measured results on this specific frozen benchmark, not a general claim about either system outside the evaluated dataset.

---

# Error Analysis

The Local run produced four valid but incorrect predictions.

| Dataset ID | Task | Expected | Local prediction |
| --- | --- | --- | --- |
| `snips_test_014` | `intent` | `SearchCreativeWork` | `SearchScreeningEvent` |
| `snips_test_010` | `mentions_location` | `true` | `false` |
| `snips_test_223` | `mentions_location` | `true` | `false` |
| `snips_test_368` | `mentions_location` | `true` | `false` |

Summary:

```text
1 intent confusion
3 location false negatives
0 location false positives
0 system failures
```

The result artifacts establish the observed predictions, but they do not establish model-internal causes for those errors.

---

# Confidence Interpretation

Confidence values are intentionally kept separate by source.

## Jev Choice

Uses the native confidence field observed in the Jev response.

Formal benchmark:

```text
average confidence on correct predictions:
0.9914285714285714
```

## Jev Noul

No separate native Jev Noul confidence field was observed.

Therefore:

```text
native_confidence = null
```

No synthetic Jev confidence is added.

## Local Choice

The local Choice confidence is the maximum normalized probability after semantic validation.

Formal benchmark:

```text
average confidence, correct predictions:   1.0
average confidence, incorrect predictions: 1.0
```

## Local Noul

The local Noul confidence is derived as:

```text
max(p, 1 - p)
```

Formal benchmark:

```text
average confidence, correct predictions:   0.948
average confidence, incorrect predictions: 0.8666666666666667
```

These values are **not presented as calibrated probabilities**.

---

# Jev Compatibility

The implementation targets the observed core typed-evaluation contract rather than attempting to reproduce every Jev field.

## Observed Jev Choice response

Fields:

- `type`
- `choice`
- `probabilities`
- `confidence`

## Observed Jev Noul response

Fields:

- `type`
- `noul`

## Other observed behavior

- multiple named questions can share one state,
- missing `state` returns HTTP `422`,
- Jev successful responses include top-level model information and usage metadata.

## Intentional local differences

The local implementation adds:

- Noul boolean `answer`,
- Noul derived `confidence`.

The local implementation does not reproduce:

- Jev `usage` metadata.

See:

```text
docs/jev_compatibility.md
```

for the captured compatibility notes.

---

# Reproducing the Benchmark

The formal benchmark is already stored under `results/`. The following commands reproduce the workflow and make real network/provider calls.

## 1. Run Jev

```bash
RUN_LIVE_TESTS=1 \
python script/run_jev.py \
  --live \
  --output results/jev_results.json
```

Expected full-run shape:

```text
28 examples
56 canonical records
```

## 2. Start the Local service

In one terminal:

```bash
python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000
```

## 3. Run Local evaluation

In another terminal:

```bash
RUN_LIVE_TESTS=1 \
python script/run_local.py \
  --live \
  --output results/local_results.json
```

## 4. Compare offline

```bash
python script/compare_results.py \
  --dataset data/evaluation_dataset.json \
  --tasks data/evaluation_tasks.json \
  --jev results/jev_results.json \
  --local results/local_results.json \
  --output results/metrics.json
```

The comparison step performs no model/API calls.

---

# Testing

## Offline suite

Run:

```bash
python -m pytest -q -m "not live"
```

Latest verified result:

```text
179 passed
0 failed
2 live tests deselected
```

There is one known third-party Starlette/AnyIO deprecation warning that does not originate from project logic.

## Dependency validation

```bash
python -m pip check
```

Latest verified result:

```text
No broken requirements found.
```

## Optional live model tests

Live tests are opt-in because they make provider calls.

```bash
RUN_LIVE_TESTS=1 \
python -m pytest tests/test_live_model.py -q
```

---

# Result Artifacts

The formal benchmark artifacts are committed under:

```text
results/jev_results.json
results/local_results.json
results/metrics.json
```

## `jev_results.json`

Contains:

- benchmark metadata,
- dataset/task hashes,
- Jev model metadata,
- one canonical record per dataset/question pair,
- raw typed answers,
- latency,
- failure records if applicable.

## `local_results.json`

Uses the same canonical record format for the local service.

## `metrics.json`

Contains:

- validated input hashes,
- task-level metrics,
- coverage,
- failure counts,
- boolean confusion matrices,
- confidence summaries,
- pairwise outcomes,
- error examples.

The result artifacts were checked for API keys, Authorization tokens, `.env` content, and private secrets before submission preparation.

---

# Troubleshooting

## `address already in use`

If port `8000` is occupied:

```bash
lsof -i :8000
```

Inspect the process:

```bash
ps -p <PID> -o pid,ppid,command
```

Stop the conflicting process if appropriate, then restart Uvicorn.

## Live runner says `RUN_LIVE_TESTS=1` is required

Live execution is intentionally guarded.

Use:

```bash
RUN_LIVE_TESTS=1 python script/run_local.py --live ...
```

or:

```bash
RUN_LIVE_TESTS=1 python script/run_jev.py --live ...
```

## Local runner cannot connect

Confirm the server is running:

```bash
curl --max-time 5 \
  -s http://127.0.0.1:8000/openapi.json
```

The OpenAPI document should contain:

```text
/v1/systemone
```

## Missing API key

Check your local `.env` file.

Do not put real keys in:

- README files,
- `.env.example`,
- committed source,
- test fixtures,
- result artifacts.

## HTTP 422

Check:

- required `state`,
- local `model` value,
- question type,
- Choice criteria count and descriptions,
- Noul criteria shape,
- unknown top-level fields.

---

# Security Notes

- `.env` must remain untracked.
- API keys must never be written to result artifacts or source files.
- Public errors intentionally avoid returning raw provider error bodies.
- The service is stateless and does not persist submitted state.
- When using the OpenAI-backed evaluator, request state and question text are sent to the configured provider. Do not submit sensitive data unless that is compatible with your provider and data-handling requirements.

---

# Limitations

- The benchmark contains only **28 examples**.
- It covers only **7 SNIPS intent classes**.
- `mentions_location` is a slot-derived boolean task rather than a separately human-authored label.
- The benchmark does not represent broad-domain performance.
- General-purpose model behavior can vary across models and future provider versions.
- Local confidence values are heuristic and are not calibrated.
- Jev Noul did not expose a separate native confidence field in the observed contract.
- **Score-type questions are not implemented.**
- The local service does not reproduce Jev usage metadata.
- Only the compatibility behavior exercised by this repository is claimed.

---

# Documentation

Additional engineering documentation:

- `docs/jev_compatibility.md` — observed Jev request/response behavior and intentional local differences.
- `docs/DECISIONS.md` — frozen behavior, validation rules, thresholds, and evaluation decisions.
- `docs/jev_service_architecture_plan.md` — service architecture and implementation plan.

---

# Dataset Attribution

Benchmark source:

[`bkonkle/snips-joint-intent`](https://huggingface.co/datasets/bkonkle/snips-joint-intent)

The project uses SNIPS labels as the source of expected intent values and derives the boolean location task from slot annotations.

The frozen benchmark data and task definitions used for the reported results are stored in:

```text
data/evaluation_dataset.json
data/evaluation_tasks.json
```

---

# License

This repository does not currently declare an open-source license.

If the project is intended for public reuse or redistribution, add an explicit `LICENSE` file that reflects the desired terms.
