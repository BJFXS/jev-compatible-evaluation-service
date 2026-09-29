import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping

import pytest

from app.config import get_settings
from script.probe_jev import build_probe_cases, run_probe, write_artifact


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = PROJECT_ROOT / "tests/fixtures/jev_contract/probe_dry_run.json"
FORBIDDEN_REQUEST_KEYS = frozenset({"expected", "source_labels", "source", "dataset_id", "authorization"})
SENSITIVE_KEYS = frozenset({"authorization", "api_key", "apikey", "access_token", "token", "secret"})


def assert_request_has_no_evaluation_or_secret_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        assert not {str(key).lower() for key in value} & FORBIDDEN_REQUEST_KEYS
        for nested in value.values():
            assert_request_has_no_evaluation_or_secret_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_request_has_no_evaluation_or_secret_fields(nested)


def assert_artifact_has_no_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        assert not {str(key).lower() for key in value} & SENSITIVE_KEYS
        for nested in value.values():
            assert_artifact_has_no_secrets(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_artifact_has_no_secrets(nested)


def stable_artifact_fields(artifact: dict[str, Any]) -> dict[str, Any]:
    """Drop only timestamps, which are intentionally generated at runtime."""
    return {
        "metadata": artifact["metadata"],
        "records": [
            {key: value for key, value in record.items() if key != "timestamp"}
            for record in artifact["records"]
        ],
    }


def test_probe_cases_are_synthetic_and_cover_the_four_contract_cases() -> None:
    cases = build_probe_cases("jev-test")
    case_by_name = {case["name"]: case for case in cases}

    assert [case["name"] for case in cases] == ["choice", "noul", "multiple_questions", "missing_state"]
    for case in cases:
        assert_request_has_no_evaluation_or_secret_fields(case["request"])

    choice_request = case_by_name["choice"]["request"]
    assert set(choice_request) == {"state", "model", "questions"}
    assert choice_request["model"] == "jev-test"
    assert len(choice_request["questions"]) == 1
    choice_question = next(iter(choice_request["questions"].values()))
    assert choice_question["type"] == "choice"
    assert isinstance(choice_question["instructions"], str) and choice_question["instructions"].strip()
    assert isinstance(choice_question["criteria"], dict) and len(choice_question["criteria"]) >= 2
    assert all(isinstance(key, str) and key.strip() for key in choice_question["criteria"])
    assert all(isinstance(value, str) and value.strip() for value in choice_question["criteria"].values())

    noul_request = case_by_name["noul"]["request"]
    assert set(noul_request) == {"state", "model", "questions"}
    assert noul_request["model"] == "jev-test"
    assert len(noul_request["questions"]) == 1
    noul_question = next(iter(noul_request["questions"].values()))
    assert noul_question["type"] == "noul"
    assert isinstance(noul_question["instructions"], str) and noul_question["instructions"].strip()
    assert set(noul_question["criteria"]) == {"true", "false"}
    assert all(isinstance(value, str) and value.strip() for value in noul_question["criteria"].values())

    multiple_request = case_by_name["multiple_questions"]["request"]
    assert set(multiple_request) == {"state", "model", "questions"}
    assert multiple_request["model"] == "jev-test"
    assert len(multiple_request["questions"]) == 2
    assert {question["type"] for question in multiple_request["questions"].values()} == {"choice", "noul"}
    assert all("state" not in question for question in multiple_request["questions"].values())

    malformed_request = case_by_name["missing_state"]["request"]
    assert "state" not in malformed_request
    assert set(malformed_request) == {"model", "questions"}
    assert malformed_request["model"] == "jev-test"
    assert malformed_request["questions"]


def test_dry_run_never_calls_sender_and_records_all_cases() -> None:
    def forbidden_sender(*_args: object) -> tuple[int, object]:
        raise AssertionError("dry-run must not access the network")

    artifact = run_probe(get_settings({}), live=False, sender=forbidden_sender)

    assert artifact["metadata"]["mode"] == "dry-run"
    assert len(artifact["records"]) == 4
    assert all(record["http_status"] is None for record in artifact["records"])


def test_live_requires_both_explicit_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_LIVE_TESTS", raising=False)

    with pytest.raises(RuntimeError, match="--live"):
        run_probe(get_settings({}), live=True)


def test_live_timeout_is_safe_and_timeout_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    seen: list[float] = []

    def timing_out(_url: str, _body: object, _key: str, timeout: float) -> tuple[int, object]:
        seen.append(timeout)
        raise socket.timeout()

    artifact = run_probe(get_settings({"JEV_API_KEY": "test-secret", "MODEL_TIMEOUT_SECONDS": "7"}), live=True, sender=timing_out)

    assert seen == [7.0] * 4
    assert all(record["error"] == "request timed out" for record in artifact["records"])


def test_artifact_redacts_request_and_response_secrets(tmp_path) -> None:
    artifact = {
        "records": [{"request_body": {"Authorization": "Bearer nope"}, "response_body": {"api_key": "nope", "ok": True}}]
    }
    destination = tmp_path / "artifact.json"
    write_artifact(destination, artifact)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    rendered = destination.read_text(encoding="utf-8")
    assert saved["records"][0]["request_body"]["Authorization"] == "[REDACTED]"
    assert saved["records"][0]["response_body"]["api_key"] == "[REDACTED]"
    assert "nope" not in rendered


def test_cli_dry_run_writes_a_clean_four_case_artifact_without_credentials(tmp_path) -> None:
    destination = tmp_path / "cli-dry-run.json"
    environment = {"PATH": os.defpath, "JEV_BASE_URL": "http://127.0.0.1:1"}

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "script/probe_jev.py"), "--output", str(destination)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(destination.read_text(encoding="utf-8"))
    assert artifact["metadata"]["mode"] == "dry-run"
    assert [record["name"] for record in artifact["records"]] == [
        "choice", "noul", "multiple_questions", "missing_state"
    ]
    assert all(record["http_status"] is None for record in artifact["records"])
    assert all(record["response_body"] is None and record["error"] is None for record in artifact["records"])
    assert_artifact_has_no_secrets(artifact)


def test_committed_fixture_matches_stable_production_dry_run_contract() -> None:
    def forbidden_sender(*_args: object) -> tuple[int, object]:
        raise AssertionError("fixture comparison dry-run must not access the network")

    generated = run_probe(get_settings({}), live=False, sender=forbidden_sender)
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert stable_artifact_fields(fixture) == stable_artifact_fields(generated)
