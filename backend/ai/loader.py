from typing import Optional, BinaryIO
from pathlib import Path
from pypdf import PdfReader


def load_document(
    document_text: Optional[str] = None,
    file: Optional[BinaryIO] = None
) -> str:
    """
    Load document text from raw text or PDF file.
    """

    if document_text:
        text = document_text.strip()
        if not text:
            raise ValueError("Provided document text is empty.")
        return _normalize_text(text)

    if file:
        try:
            reader = PdfReader(file)
            pages = []

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)

            if not pages:
                raise ValueError("No readable text found in PDF.")

            return _normalize_text("\n".join(pages))

        except Exception as e:
            raise ValueError(f"Failed to load PDF: {e}")

    raise ValueError("No document text or PDF file provided.")


def load_document_from_path(file_path: str) -> str:
    """Load document text from a file path."""
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    with open(path, 'rb') as f:
        return load_document(file=f)


def _normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return " ".join(text.split())
