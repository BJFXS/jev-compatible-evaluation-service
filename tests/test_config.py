from pathlib import Path

import app.config as config
from app.config import get_settings


def test_missing_dotenv_and_environment_use_defaults_without_credentials(tmp_path: Path) -> None:
    settings = get_settings({}, dotenv_path=tmp_path / "missing.env")

    assert settings.openai_api_key is None
    assert settings.jev_api_key is None
    assert settings.jev_model == "jev-latest"
    assert settings.model_timeout_seconds == 30.0
    assert settings.max_model_retries == 1
    assert settings.probability_sum_tolerance == 0.01
    assert settings.noul_boolean_threshold == 0.5
    assert settings.max_questions == 10


def test_temporary_dotenv_loads_supported_values_and_converts_numbers(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join([
            "OPENAI_API_KEY=dotenv-openai", "OPENAI_BASE_URL=https://openai.example/v1",
            "OPENAI_MODEL=dotenv-model", "JEV_API_KEY=dotenv-jev", "JEV_MODEL=dotenv-jev-model",
            "JEV_BASE_URL=https://jev.example/", "LOCAL_BASE_URL=http://127.0.0.1:9999/",
            "MAX_MODEL_RETRIES=2", "MODEL_TIMEOUT_SECONDS=12.5", "PROBABILITY_SUM_TOLERANCE=0.02",
            "NOUL_BOOLEAN_THRESHOLD=0.6", "MAX_QUESTIONS=4", "LOG_LEVEL=DEBUG",
        ]) + "\n",
        encoding="utf-8",
    )
    settings = get_settings({}, dotenv_path=dotenv_path)

    assert settings.openai_api_key == "dotenv-openai"
    assert settings.openai_base_url == "https://openai.example/v1"
    assert settings.openai_model == "dotenv-model"
    assert settings.jev_api_key == "dotenv-jev"
    assert settings.jev_model == "dotenv-jev-model"
    assert settings.jev_base_url == "https://jev.example"
    assert settings.local_base_url == "http://127.0.0.1:9999"
    assert settings.model_timeout_seconds == 12.5
    assert settings.max_model_retries == 2
    assert settings.probability_sum_tolerance == 0.02
    assert settings.noul_boolean_threshold == 0.6
    assert settings.max_questions == 4
    assert settings.log_level == "DEBUG"


def test_process_environment_wins_over_temporary_dotenv(tmp_path: Path, monkeypatch) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "JEV_API_KEY=dotenv-credential\nJEV_MODEL=dotenv-model\nMODEL_TIMEOUT_SECONDS=9\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("JEV_API_KEY", "process-credential")
    monkeypatch.setenv("JEV_MODEL", "process-model")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "7")
    settings = get_settings(dotenv_path=dotenv_path)

    assert settings.jev_api_key == "process-credential"
    assert settings.jev_model == "process-model"
    assert settings.model_timeout_seconds == 7.0


def test_project_dotenv_path_is_stable_from_any_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert config.PROJECT_DOTENV_PATH == Path(config.__file__).resolve().parent.parent / ".env"
