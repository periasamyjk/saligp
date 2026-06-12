"""
Document text extraction utilities for SALIGP.

The classifier still consumes fixed numeric features; this module turns common
document inputs into text records before feature computation.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree


class UnsupportedDocumentError(ValueError):
    """Raised when a file type cannot be converted to text."""


@dataclass(frozen=True)
class DocumentRecord:
    """Text plus lightweight metadata used for pairwise deduplication."""

    document_id: str
    filename: str
    text: str
    size_bytes: int = 0
    content_type: Optional[str] = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


class DocumentProcessor:
    """Extract text from txt, md, html, json, csv, docx, and pdf files."""

    TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rtf", ".log"}
    HTML_EXTENSIONS = {".html", ".htm"}
    JSON_EXTENSIONS = {".json", ".jsonl"}
    CSV_EXTENSIONS = {".csv", ".tsv"}

    def from_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> DocumentRecord:
        suffix = Path(filename).suffix.lower()
        text = self.extract_text(content, filename)
        return DocumentRecord(
            document_id=document_id or filename,
            filename=filename,
            text=text,
            size_bytes=len(content),
            content_type=content_type,
        )

    def from_path(self, path: Path) -> DocumentRecord:
        content = path.read_bytes()
        return self.from_bytes(content, path.name, document_id=str(path))

    def extract_text(self, content: bytes, filename: str) -> str:
        suffix = Path(filename).suffix.lower()

        if suffix in self.TEXT_EXTENSIONS:
            return self._decode_text(content)
        if suffix in self.HTML_EXTENSIONS:
            return self._extract_html(content)
        if suffix in self.JSON_EXTENSIONS:
            return self._extract_json(content)
        if suffix in self.CSV_EXTENSIONS:
            return self._extract_csv_text(content, suffix)
        if suffix == ".docx":
            return self._extract_docx(content)
        if suffix == ".pdf":
            return self._extract_pdf(content)

        # Last-chance decode for extensionless text uploads.
        decoded = self._decode_text(content)
        if self._looks_like_text(decoded):
            return decoded

        raise UnsupportedDocumentError(
            f"Unsupported document type '{suffix or 'unknown'}'. "
            "Supported: txt, md, html, json, csv, tsv, docx, pdf."
        )

    def _decode_text(self, content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return content.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore").strip()

    def _extract_html(self, content: bytes) -> str:
        parser = _HTMLTextExtractor()
        parser.feed(self._decode_text(content))
        return parser.get_text().strip()

    def _extract_json(self, content: bytes) -> str:
        raw = self._decode_text(content)
        if not raw:
            return ""
        if raw.lstrip().startswith("{") or raw.lstrip().startswith("["):
            return self._flatten_json(json.loads(raw))
        return " ".join(self._flatten_json(json.loads(line)) for line in raw.splitlines() if line.strip())

    def _flatten_json(self, value) -> str:
        if isinstance(value, dict):
            return " ".join(self._flatten_json(v) for v in value.values())
        if isinstance(value, list):
            return " ".join(self._flatten_json(v) for v in value)
        if value is None:
            return ""
        return str(value)

    def _extract_csv_text(self, content: bytes, suffix: str) -> str:
        delimiter = "\t" if suffix == ".tsv" else ","
        text = self._decode_text(content)
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        cells: List[str] = []
        for row in reader:
            cells.extend(cell for cell in row if cell.strip())
        return " ".join(cells).strip()

    def _extract_docx(self, content: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as docx:
                xml_bytes = docx.read("word/document.xml")
        except (KeyError, zipfile.BadZipFile) as exc:
            raise UnsupportedDocumentError("Unable to read DOCX document text.") from exc

        root = ElementTree.fromstring(xml_bytes)
        paragraphs: List[str] = []
        for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            text = "".join(
                node.text or ""
                for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            )
            if text.strip():
                paragraphs.append(text.strip())
        return "\n".join(paragraphs)

    def _extract_pdf(self, content: bytes) -> str:
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader = module.PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages).strip()
                if text:
                    return text
            except Exception:
                continue

        decoded = self._decode_text(content)
        cleaned = re.sub(r"\s+", " ", decoded)
        if self._looks_like_text(cleaned):
            return cleaned.strip()

        raise UnsupportedDocumentError(
            "PDF text extraction requires an extractable-text PDF and pypdf/PyPDF2 installed."
        )

    def _looks_like_text(self, text: str) -> bool:
        if not text:
            return False
        printable = sum(ch.isprintable() or ch.isspace() for ch in text)
        return printable / max(len(text), 1) > 0.9

