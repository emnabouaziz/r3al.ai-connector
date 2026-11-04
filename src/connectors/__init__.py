"""Source connectors. Concrete connectors register themselves via ``src.registry.register``."""

from src.connectors.base import SourceConnector

__all__ = ["SourceConnector"]
