# Jev-Compatible Typed Evaluation Service

## Project Overview

This project implements a Jev-compatible typed evaluation service using a general-purpose GPT model for Choice and Noul evaluation. It evaluates the local service and Jev on a frozen, labeled SNIPS benchmark.

## Assignment Scope

- Build a local Jev-like HTTP service with typed requests and responses.
- Create a labeled benchmark for intent classification and a boolean task.
- Run the same benchmark through Jev and the local service, then compare the recorded results offline.

## Dataset

Source: [`bkonkle/snips-joint-intent`](https://huggingface.co/datasets/bkonkle/snips-joint-intent)

The frozen evaluation set contains 28 examples: 7 intents × 4 examples.

- Intent ground truth comes from the SNIPS intent label.
- `mentions_location` is derived from location-related slot annotations.
- Jev output is never used as ground truth.

Frozen input hashes:

- Dataset: `28c4f369a01637a1bf123a6930dfbccd9e6c0015cd31051fb8ed857ab891ea2d`
- Tasks: `299f0e3d8450a11debef72eba000c6d33e1183e35b870be83d58c02f36bb838b`

## Evaluation Tasks

- **Choice:** “What is the user's intent?”
- **Noul:** “Does the utterance explicitly mention a location?”

## Architecture

The service uses four layers:

1. **Entry Layer** — `app/main.py` provides FastAPI request validation, response serialization, and safe HTTP error mapping.
2. **Service Layer** — `app/service.py` performs sequential named-question orchestration and owns the retry budget.
3. **Model Layer** — `app/model.py` contains the provider-neutral evaluator interface and the GPT Structured Outputs adapter.
4. **Normalization Layer** — `app/normalization.py` performs pure semantic validation and deterministic typed-answer conversion.

The Model layer owns provider calls, the Service layer owns retry, and Normalization is pure. The generic `app/` service contains no dataset-specific labels, IDs, or expected answers.

## Jev Compatibility

Observed Jev native response fields:

- Choice: `type`, `choice`, `probabilities`, `confidence`
- Noul: `type`, `noul`

Observed request behavior:

- Multiple named questions sharing one state are supported.
- A missing `state` receives HTTP 422.

Intentional local differences:

- Local Noul adds a boolean `answer`.
- Local Noul adds derived heuristic `confidence`.
- The local response omits Jev `usage` metadata.

See `docs/jev_compatibility.md` for observed Jev facts and `docs/DECISIONS.md` for frozen local behavior.

## Reliability

- OpenAI Structured Outputs constrain model-produced fields.
- Pydantic validates request and typed model-output shapes.
- Normalization performs semantic validation, including probability-key and argmax checks.
- The Service layer controls retry; OpenAI SDK automatic retries are disabled.
- Error responses use safe public messages rather than raw provider details.
- The service is stateless.

## Setup

Use Python 3.12.7 in the dedicated `jev-hw1` environment.

```bash
python -m pip install -r requirements.txt
```

## Environment

Use `.env.example` as the reference. Never place real secrets, tokens, or Authorization values in tracked files.

Supported variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `JEV_API_KEY`
- `JEV_MODEL`
- `JEV_BASE_URL`
- `LOCAL_BASE_URL`
- `MAX_MODEL_RETRIES`
- `MODEL_TIMEOUT_SECONDS`
- `PROBABILITY_SUM_TOLERANCE`
- `NOUL_BOOLEAN_THRESHOLD`
- `MAX_QUESTIONS`
- `LOG_LEVEL`

## Run Local Service

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Tests

```bash
python -m pytest -q -m "not live"
```

## Run Jev Evaluation

```bash
RUN_LIVE_TESTS=1 \
python script/run_jev.py \
  --live \
  --output results/jev_results.json
```

## Run Local Evaluation

The local FastAPI service must already be running.

```bash
RUN_LIVE_TESTS=1 \
python script/run_local.py \
  --live \
  --output results/local_results.json
```

## Compare Results

```bash
python script/compare_results.py \
  --dataset data/evaluation_dataset.json \
  --tasks data/evaluation_tasks.json \
  --jev results/jev_results.json \
  --local results/local_results.json \
  --output results/metrics.json
```

## Final Benchmark Results

The completed frozen benchmark produced the following factual results.

| Service | Task | Correct | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Jev | intent | 28/28 | 1.0 | — | — | — |
| Jev | mentions_location | 28/28 | 1.0 | 1.0 | 1.0 | 1.0 |
| Local | intent | 27/28 | 0.9642857142857143 | — | — | — |
| Local | mentions_location | 25/28 | 0.8928571428571429 | 1.0 | 0.75 | 0.8571428571428571 |

Overall results:

- Jev: 56/56, accuracy `1.0`
- Local: 52/56, accuracy `0.9285714285714286`
- Coverage: Jev `1.0`; Local `1.0`
- Failures: Jev `0`; Local `0`

These are reported as measured results, not as a subjective ranking.

## Error Analysis

The Local artifact contains four valid but incorrect predictions:

1. `snips_test_014` / `intent`: expected `SearchCreativeWork`; actual `SearchScreeningEvent`.
2. `snips_test_010` / `mentions_location`: expected `true`; actual `false`.
3. `snips_test_223` / `mentions_location`: expected `true`; actual `false`.
4. `snips_test_368` / `mentions_location`: expected `true`; actual `false`.

This is one intent confusion and three location false negatives, with no false positives and no system failures. The artifacts do not establish a claim about model-internal causes.

## Confidence Interpretation

- Jev Choice confidence is the observed native Jev field.
- No native Jev Noul confidence was observed, so it remains `null`.
- Local Choice confidence is a heuristic/model-produced normalized confidence.
- Local Noul confidence is a derived heuristic confidence.

None of these values are presented as calibrated probabilities.

## Limitations

- The benchmark has only 28 examples.
- It covers only 7 SNIPS intent classes.
- The Noul task is a slot-derived boolean label.
- General-purpose LLM behavior may vary across runs and models.
- Local confidence is not calibrated.
- Score-type questions are out of scope.
- The local API intentionally does not reproduce Jev `usage` metadata.
- These benchmark results do not establish performance in broader domains.

## Result Artifacts

- `results/jev_results.json`
- `results/local_results.json`
- `results/metrics.json`

These artifacts contain the completed benchmark records and metrics. They were checked for API keys, Authorization tokens, `.env` content, and private secrets before submission preparation.
