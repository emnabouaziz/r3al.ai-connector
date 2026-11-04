from collections.abc import Iterator

import pytest

from src import registry
from src.connectors.base import SourceConnector
from src.contracts import ItemRef, RawDocument
from src.registry import DuplicateSourceType, UnknownSourceType, build, register


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "CONNECTORS", {})


def _make_connector(source_type: str) -> type[SourceConnector]:
    class Dummy(SourceConnector):
        def test_connection(self) -> bool:
            return True

        def list_items(self) -> Iterator[ItemRef]:
            return iter(())

        def fetch(self, ref: ItemRef) -> RawDocument:
            raise NotImplementedError

    Dummy.source_type = source_type
    return Dummy


def test_register_rejects_duplicate_source_type() -> None:
    register(_make_connector("dummy"))
    with pytest.raises(DuplicateSourceType):
        register(_make_connector("dummy"))


def test_build_raises_unknown_source_type_listing_known() -> None:
    register(_make_connector("sql"))
    register(_make_connector("files"))
    with pytest.raises(UnknownSourceType) as exc:
        build({"name": "x", "type": "email", "config": {}})
    assert exc.value.known == ["files", "sql"]
    assert "files" in str(exc.value) and "sql" in str(exc.value)


def test_build_instantiates_connector_with_name_and_config() -> None:
    register(_make_connector("sql"))
    connector = build({"name": "crm_postgres", "type": "sql", "config": {"dsn": "x"}})
    assert connector.name == "crm_postgres"
    assert connector.config == {"dsn": "x"}
    assert connector.test_connection()
