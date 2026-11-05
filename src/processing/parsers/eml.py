"""Single .eml message -> text.

Prefers the text/plain part; falls back to degrading text/html (tags out,
block elements to line breaks, entities unescaped); returns "" when neither
yields text (the pipeline drops that with 'eml_no_text_part'). Charsets are
decoded from the MIME headers, then repaired with ftfy.
"""

import email
from email import policy
from email.message import EmailMessage, Message
from html.parser import HTMLParser

import ftfy

_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table", "blockquote"}


class _HtmlToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("style", "script", "head"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("style", "script", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def parse_message(content: bytes) -> EmailMessage:
    message = email.message_from_bytes(content, policy=policy.default)
    assert isinstance(message, EmailMessage)
    return message


def _part_text(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:  # unknown charset label
        return payload.decode("utf-8", errors="replace")


def message_text(message: EmailMessage) -> str:
    plain = message.get_body(preferencelist=("plain",))
    if plain is not None:
        text = _part_text(plain).strip()
        if text:
            return ftfy.fix_text(text)
    html = message.get_body(preferencelist=("html",))
    if html is not None:
        text = html_to_text(_part_text(html))
        if text:
            return ftfy.fix_text(text)
    return ""


def extract_text(content: bytes) -> str:
    return message_text(parse_message(content))
