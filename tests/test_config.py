import os
from pathlib import Path

import pytest

from src.config import MissingEnvVar, expand_env, load_config


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_resolves_var_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DSN", "postgresql://u:p@h/db")
    cfg = load_config(_write(tmp_path, "sources:\n  - config:\n      dsn: ${TEST_DSN}\n"))
    assert cfg["sources"][0]["config"]["dsn"] == "postgresql://u:p@h/db"


def test_applies_default_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_DOCS_DIR", raising=False)
    cfg = load_config(_write(tmp_path, "root: ${TEST_DOCS_DIR:-./data/documents}\n"))
    assert cfg["root"] == "./data/documents"


def test_env_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DOCS_DIR", "/srv/docs")
    cfg = load_config(_write(tmp_path, "root: ${TEST_DOCS_DIR:-./data/documents}\n"))
    assert cfg["root"] == "/srv/docs"


def test_raises_missing_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_ABSENT", raising=False)
    with pytest.raises(MissingEnvVar) as exc:
        load_config(_write(tmp_path, "dsn: ${TEST_ABSENT}\n"))
    assert exc.value.name == "TEST_ABSENT"


def test_expansion_is_recursive_and_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_HOST", "db")
    data = {"a": ["x-${TEST_HOST}-y", {"b": "${TEST_HOST}"}], "n": 3}
    assert expand_env(data) == {"a": ["x-db-y", {"b": "db"}], "n": 3}


def test_dotenv_next_to_config_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_FROM_DOTENV", raising=False)
    (tmp_path / ".env").write_text("TEST_FROM_DOTENV=hello\n", encoding="utf-8")
    try:
        cfg = load_config(_write(tmp_path, "v: ${TEST_FROM_DOTENV}\n"))
        assert cfg["v"] == "hello"
    finally:
        os.environ.pop("TEST_FROM_DOTENV", None)


def test_demo_config_uses_fake_source() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "config.yaml"))
    assert [s["type"] for s in cfg["sources"]] == ["fake"]


def test_production_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://dummy")
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "config.production.yaml"))
    assert [s["type"] for s in cfg["sources"]] == ["sql", "files", "email"]
    assert cfg["sources"][0]["config"]["dsn"] == "postgresql://dummy"
