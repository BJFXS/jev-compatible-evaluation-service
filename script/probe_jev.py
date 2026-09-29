"""Guarded, synthetic compatibility probe for the reference Jev service."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# Permit both ``python -m script.probe_jev`` and the documented script path form.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings


JsonObject = dict[str, Any]
Sender = Callable[[str, JsonObject, str, float], tuple[int, Any]]
SENSITIVE_KEYS = frozenset({"authorization", "api_key", "apikey", "access_token", "token", "secret"})


def build_probe_cases(model: str) -> list[JsonObject]:
    """Build four non-dataset requests, including one intentionally invalid body."""
    choice = {
        "state": "Please arrange a table for two this evening.",
        "model": model,
        "questions": {
            "routing": {
                "type": "choice",
                "instructions": "Choose the request category.",
                "criteria": {
                    "reserve_table": "The user wants to reserve a table.",
                    "check_hours": "The user asks for opening hours.",
                },
            }
        },
    }
    noul = {
        "state": {"message": "Is the library open near Union Square?"},
        "model": model,
        "questions": {
            "mentions_place": {
                "type": "noul",
                "instructions": "Does the message explicitly mention a place?",
                "criteria": {
                    "true": "It explicitly names or refers to a place.",
                    "false": "It does not explicitly name or refer to a place.",
                },
            }
        },
    }
    multiple = {
        "state": ["Could you tell me the opening hours for Harbor Cafe?"],
        "model": model,
        "questions": {
            "routing": {
                "type": "choice",
                "instructions": "Choose the request category.",
                "criteria": {
                    "reserve_table": "The user wants to reserve a table.",
                    "check_hours": "The user asks for opening hours.",
                },
            },
            "mentions_place": {
                "type": "noul",
                "instructions": "Does the message explicitly mention a place?",
                "criteria": {
                    "true": "It explicitly names or refers to a place.",
                    "false": "It does not explicitly name or refer to a place.",
                },
            },
        },
    }
    missing_state = {
        "model": model,
        "questions": choice["questions"],
    }
    return [
        {"name": "choice", "request": choice, "expected_kind": "success"},
        {"name": "noul", "request": noul, "expected_kind": "success"},
        {"name": "multiple_questions", "request": multiple, "expected_kind": "success"},
        {"name": "missing_state", "request": missing_state, "expected_kind": "client_error"},
    ]


def redact(value: Any) -> Any:
    """Remove credentials from artifacts, including a service that echoes them."""
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _response_body(response: Any) -> Any:
    raw = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def send_request(url: str, body: JsonObject, api_key: str, timeout: float) -> tuple[int, Any]:
    """POST directly to Jev; this intentionally does not use the local service."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- explicit opt-in live probe
            return response.status, _response_body(response)
    except HTTPError as error:
        return error.code, _response_body(error)


def _safe_error(error: Exception) -> str:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "request timed out"
    return f"{type(error).__name__}: request failed"


def run_probe(settings: Settings, *, live: bool, sender: Sender = send_request) -> JsonObject:
    """Run the guarded probe and return a redacted, serializable artifact."""
    enabled = live and os.environ.get("RUN_LIVE_TESTS") == "1"
    if live and not enabled:
        raise RuntimeError("Live probes require both --live and RUN_LIVE_TESTS=1.")
    if enabled and not settings.jev_api_key:
        raise RuntimeError("JEV_API_KEY is required for a live probe.")

    source = settings.jev_base_url if enabled else "synthetic dry-run (no network)"
    records: list[JsonObject] = []
    for case in build_probe_cases(settings.jev_model):
        record: JsonObject = {
            "name": case["name"],
            "expected_kind": case["expected_kind"],
            "request_body": redact(case["request"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "http_status": None,
            "response_body": None,
            "error": None,
        }
        if enabled:
            try:
                status, response = sender(
                    f"{settings.jev_base_url}/v1/systemone",
                    case["request"],
                    settings.jev_api_key or "",
                    settings.model_timeout_seconds,
                )
                record["http_status"] = status
                record["response_body"] = redact(response)
            except Exception as error:  # artifact must retain a safe probe failure
                record["error"] = _safe_error(error)
        records.append(record)
    return {"metadata": {"mode": "live" if enabled else "dry-run", "source": source}, "records": records}


def write_artifact(path: Path, artifact: JsonObject) -> None:
    """Atomically write only redacted probe data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(redact(artifact), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="send guarded requests to the reference service")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/jev_contract/probe_dry_run.json"))
    args = parser.parse_args()
    artifact = run_probe(get_settings(), live=args.live)
    write_artifact(args.output, artifact)
    print(f"Wrote {len(artifact['records'])} {artifact['metadata']['mode']} probe records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
