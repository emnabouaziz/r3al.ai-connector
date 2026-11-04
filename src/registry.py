"""Registry mapping a ``source_type`` to its connector class."""

import logging
from typing import Any, TypeVar

from src.connectors.base import SourceConnector

logger = logging.getLogger(__name__)

CONNECTORS: dict[str, type[SourceConnector]] = {}

C = TypeVar("C", bound=type[SourceConnector])


class RegistryError(Exception):
    """Base class for registry errors."""


class DuplicateSourceType(RegistryError):
    """Raised when two connectors claim the same ``source_type``."""

    def __init__(self, source_type: str, existing: str) -> None:
        self.source_type = source_type
        super().__init__(f"source_type '{source_type}' is already registered by {existing}")


class UnknownSourceType(RegistryError):
    """Raised by ``build`` when no connector is registered for a type."""

    def __init__(self, source_type: str, known: list[str]) -> None:
        self.source_type = source_type
        self.known = known
        known_str = ", ".join(known) if known else "<none>"
        super().__init__(f"unknown source_type '{source_type}'; known types: {known_str}")


class InvalidSourceConfig(RegistryError):
    """Raised when a source entry from config is malformed."""


def register(cls: C) -> C:
    """Class decorator registering a connector under its ``source_type``."""
    source_type = getattr(cls, "source_type", None)
    if not isinstance(source_type, str) or not source_type:
        raise InvalidSourceConfig(f"{cls.__name__} must define a non-empty 'source_type'")
    existing = CONNECTORS.get(source_type)
    if existing is not None and existing is not cls:
        raise DuplicateSourceType(source_type, existing.__qualname__)
    CONNECTORS[source_type] = cls
    logger.debug("connector_registered source_type=%s cls=%s", source_type, cls.__qualname__)
    return cls


def build(cfg: dict[str, Any]) -> SourceConnector:
    """cfg = {"name":..., "type":..., "config":{...}}.
    Raises UnknownSourceType listing known types when absent."""
    for key in ("name", "type"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise InvalidSourceConfig(f"source entry is missing a non-empty '{key}'")
    connector_config = cfg.get("config", {})
    if not isinstance(connector_config, dict):
        raise InvalidSourceConfig(f"source '{cfg['name']}': 'config' must be a mapping")

    source_type: str = cfg["type"]
    cls = CONNECTORS.get(source_type)
    if cls is None:
        raise UnknownSourceType(source_type, sorted(CONNECTORS))

    connector = cls(connector_config)
    connector.name = cfg["name"]
    logger.info("connector_built name=%s source_type=%s", connector.name, source_type)
    return connector
