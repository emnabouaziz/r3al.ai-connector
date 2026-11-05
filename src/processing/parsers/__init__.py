"""Parsers selected by mime_type: the parse stage's only dispatch.

Each entry says how to turn RawDocument.content into text and which drop
reason applies when that text is empty. Adding a format = adding a row; the
pipeline itself never branches on the source.
"""

from collections.abc import Callable
from dataclasses import dataclass

from src.processing.parsers import docx, pdf

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIME = "text/plain"
# Email threads: messages already decoded by the connector, joined by MESSAGE_SEPARATOR.
THREAD_CSV_MIME = "text/x-email-thread"
THREAD_EML_MIME = "message/x-email-thread"


class UnsupportedMimeType(ValueError):
    """No parser is registered for this mime_type."""


@dataclass(frozen=True)
class Parser:
    extract: Callable[[bytes | str], str]
    # Drop reason when the extracted text is empty; None lets the clean stage decide.
    empty_reason: str | None = None


def _plain_text(content: bytes | str) -> str:
    return content.decode("utf-8") if isinstance(content, bytes) else content


def _as_bytes(fn: Callable[[bytes], str]) -> Callable[[bytes | str], str]:
    def wrapper(content: bytes | str) -> str:
        return fn(content.encode("utf-8") if isinstance(content, str) else content)

    return wrapper


PARSERS: dict[str, Parser] = {
    TEXT_MIME: Parser(_plain_text),
    PDF_MIME: Parser(_as_bytes(pdf.extract_text), empty_reason="pdf_no_text_layer"),
    DOCX_MIME: Parser(_as_bytes(docx.extract_text), empty_reason="docx_no_text"),
    # An empty CSV thread is left to the clean stage ('empty_text'); an .eml thread
    # with no text/plain nor text/html part at all is dropped here.
    THREAD_CSV_MIME: Parser(_plain_text),
    THREAD_EML_MIME: Parser(_plain_text, empty_reason="eml_no_text_part"),
}

MIME_BY_EXTENSION: dict[str, str] = {".pdf": PDF_MIME, ".docx": DOCX_MIME, ".txt": TEXT_MIME}


def get_parser(mime_type: str) -> Parser:
    try:
        return PARSERS[mime_type]
    except KeyError:
        raise UnsupportedMimeType(mime_type) from None
