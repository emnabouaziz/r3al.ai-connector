"""PDF / DOCX parsers, line rejoining and chunking, on the generated documents."""

from pathlib import Path

import pytest

from src.processing.chunking import chunk_text
from src.processing.parsers import DOCX_MIME, PDF_MIME, get_parser
from src.processing.parsers.text import rejoin_broken_lines
from src.synthetic.documents import generate_documents


@pytest.fixture(scope="module")
def docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("documents")
    generate_documents(root, seed=42)
    return root


def pdf_text(path: Path) -> str:
    return get_parser(PDF_MIME).extract(path.read_bytes())


def docx_text(path: Path) -> str:
    return get_parser(DOCX_MIME).extract(path.read_bytes())


def test_generation_writes_14_reproducible_files(tmp_path: Path, docs: Path) -> None:
    files = sorted(p.name for p in docs.iterdir())
    assert len(files) == 14
    again = tmp_path / "again"
    generate_documents(again, seed=42)
    for name in files:
        assert (docs / name).read_bytes() == (again / name).read_bytes(), name


def test_scanned_pdf_yields_empty_text(docs: Path) -> None:
    assert pdf_text(docs / "internal_claims_note_scanned.pdf") == ""


def test_corrupt_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        get_parser(PDF_MIME).extract(b"%PDF-1.4 this is not a pdf")


def test_tariff_table_is_extracted(docs: Path) -> None:
    text = docx_text(docs / "tariffs_2026.docx")
    assert "Produit | Formule | Prime annuelle | Franchise" in text
    assert "Auto | Omnium complète | 845 EUR | 250 EUR" in text
    assert "Familiale | Responsabilité civile vie privée | 96 EUR | 250 EUR" in text


def test_mojibake_is_repaired_and_lines_rejoined(docs: Path) -> None:
    text = docx_text(docs / "broken_encoding_note_nl.docx")
    for word in ("geïnformeerd", "ideeën", "één", "vóór", "beëindigen"):
        assert word in text, word
    assert "Ã" not in text
    assert (
        "De klant werd op 12 maart geïnformeerd dat de schadevergoeding pas na de expertise wordt uitbetaald."
        in text
    )


def test_pdf_wrapped_lines_are_rejoined_and_sections_kept(docs: Path) -> None:
    text = pdf_text(docs / "policy_conditions_auto_fr.pdf")
    lines = text.splitlines()
    assert "4. Exclusions" in lines
    body = lines[lines.index("4. Exclusions") + 1]
    assert body.startswith("Sont exclus de la couverture") and body.endswith("prévues par la loi.")


def test_docx_heading_styles_become_markers(docs: Path) -> None:
    text = docx_text(docs / "faq_customers_nl.docx")
    assert "# Veelgestelde vragen van klanten" in text
    assert "## Hoe vraag ik een offerte aan?" in text


def test_rejoin_keeps_headings_tables_and_bullets() -> None:
    raw = "1. Scope\nThe policy covers\nyour family.\nA | B | C\n- first item\nsecond line of the item\n2. Next"
    assert rejoin_broken_lines(raw).splitlines() == [
        "1. Scope",
        "The policy covers your family.",
        "A | B | C",
        "- first item second line of the item",
        "2. Next",
    ]


def test_short_text_is_one_unchanged_chunk() -> None:
    text = "Goedemiddag,\n\nIk heb een vraag over mijn factuur.\n\nGroeten, [PERSON]."
    assert chunk_text(text, 1200) == [text]


def test_long_document_is_split_at_headings(docs: Path) -> None:
    text = pdf_text(docs / "policy_conditions_liability_en.pdf")
    chunks = chunk_text(text, 1200)
    assert len(chunks) >= 6
    assert all(len(c) <= 1200 for c in chunks)
    assert any(c.startswith("3. Exclusions") for c in chunks)
    assert not any(c.lstrip().startswith("#") for c in chunks)


def test_section_without_headings_falls_back_to_paragraph_budget() -> None:
    paragraph = "Dit is een zin over de polis en de dekking. " * 12
    text = "\n\n".join([paragraph.strip()] * 5)
    chunks = chunk_text(text, 1200)
    assert len(chunks) > 1 and all(len(c) <= 1200 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")
