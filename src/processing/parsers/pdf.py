"""PDF text extraction (pypdf).

A PDF without a text layer (a scan) yields "" - that is NOT an exception:
the pipeline drops it with reason 'pdf_no_text_layer'. OCR is out of scope
(see README, next steps). A corrupt PDF raises, and the pipeline drops that
single item.
"""

from io import BytesIO

from pypdf import PdfReader

from src.processing.parsers.text import rejoin_broken_lines


def extract_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)
    if not text.strip():
        return ""
    return rejoin_broken_lines(text)
