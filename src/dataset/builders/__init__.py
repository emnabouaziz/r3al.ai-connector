"""Example builders: ticket threads -> Q/A pairs, chunks -> grounded Q/A."""

from src.dataset.builders.base import BuildCandidate, BuildSettings, ExampleBuilder, placeholder_share
from src.dataset.builders.documents import ChunkBuilder
from src.dataset.builders.facts import FactsBuilder
from src.dataset.builders.tickets import TicketThreadBuilder

__all__ = [
    "BuildCandidate",
    "BuildSettings",
    "ChunkBuilder",
    "ExampleBuilder",
    "FactsBuilder",
    "TicketThreadBuilder",
    "placeholder_share",
]
