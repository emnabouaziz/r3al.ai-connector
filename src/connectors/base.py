"""Abstract base class for all source connectors."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar

from src.contracts import ItemRef, RawDocument

logger = logging.getLogger(__name__)


class SourceConnector(ABC):
    """A source of raw documents (sql, files, email, ...).

    Subclasses set ``source_type`` as a class attribute and are registered
    with ``src.registry.register``.
    """

    source_type: ClassVar[str]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        # Logical source name from config.yaml; set by ``registry.build``.
        self.name: str = self.source_type

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True when the source is reachable and readable."""

    @abstractmethod
    def list_items(self) -> Iterator[ItemRef]:
        """Yield a reference for every item available in the source."""

    @abstractmethod
    def fetch(self, ref: ItemRef) -> RawDocument:
        """Fetch the raw content of one item."""
