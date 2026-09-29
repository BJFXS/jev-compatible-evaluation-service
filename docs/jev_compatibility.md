# Jev compatibility record

## Documented

- The reference endpoint is documented as `POST /v1/systemone` by the Jev API documentation linked from `DECISIONS.md`.
- This project sends only `state`, `model`, and `questions` in evaluation request bodies.
- The Module 0 probe uses synthetic Choice, Noul, multiple-question, and missing-state cases. It never uses the 28 evaluation examples.

## Observed

- Choice request: HTTP 200. The top-level response keys were `model`, `answers`, and
  `usage`. Its Choice answer had `type`, `choice`, `confidence`, and
  `probabilities` fields.
- Noul request: HTTP 200. Its Noul answer had only `type` and `noul` fields. No
  separate confidence, typed boolean answer, or probabilities object was observed.
- Multiple named questions sharing one state: HTTP 200. One response contained
  answers under the original question names.
- Missing `state`: HTTP 422. The top-level error field was `detail`; each
  validation detail included `type`, `loc`, `msg`, and `input`, with `loc`
  identifying `body/state`.

## Not yet verified

- The accepted model identifiers, authentication scheme, timeout behavior, and
  the semantics or calculation of the observed Choice confidence field.
- Multi-question response ordering and failure semantics beyond the observed
  successful named-answer mapping.

## Local differences

- The planned local service accepts only `gpt-local`; Jev probes use `JEV_MODEL`, whose initial default is `jev-latest`.
- The local service intentionally adds derived Noul boolean `answer` and
  `confidence` fields for the assignment. These are not Jev native fields.
- The local service does not currently reproduce Jev `usage`.
