"""Config loader with ``${VAR}`` / ``${VAR:-default}`` environment expansion."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(Exception):
    """Base class for configuration errors."""


class MissingEnvVar(ConfigError):
    """Raised when ``${VAR}`` has no default and VAR is not set."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"environment variable '{name}' is not set and has no default")


def _expand_str(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        env_value = os.environ.get(name)
        if default is not None:
            # Shell semantics: default applies when unset OR empty.
            return env_value if env_value else default
        if env_value is None:
            raise MissingEnvVar(name)
        return env_value

    return _ENV_PATTERN.sub(replace, value)


def expand_env(value: Any) -> Any:
    """Recursively expand environment references in dicts, lists and strings."""
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and expand environment variables.

    A ``.env`` file located next to the config file is loaded first when
    present. Variables already set in the environment take precedence.
    """
    config_path = Path(path)
    env_file = config_path.parent / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
        logger.debug("dotenv_loaded")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    expanded: dict[str, Any] = expand_env(raw)
    logger.info("config_loaded sources=%d", len(expanded.get("sources", []) or []))
    return expanded


def load_taxonomy(path: str) -> dict[str, Any]:
    """Load the domain taxonomy YAML once; pass the result to ``classify_domain``."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("domains"), dict):
        raise ConfigError("taxonomy must contain a 'domains' mapping")
    taxonomy: dict[str, Any] = raw
    logger.info("taxonomy_loaded domains=%d", len(taxonomy["domains"]))
    return taxonomy
