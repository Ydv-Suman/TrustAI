"""Document loader module for processing text and PDF documents.

This module provides utilities for loading and cleaning document text from
various sources, including direct text input and PDF files.
"""

from typing import Optional
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Clean and normalize text by fixing encoding issues and whitespace.
    
    This function performs several text normalization operations:
    - Replaces carriage returns with newlines
    - Fixes common ligature encoding issues (ﬁ -> fi, ﬂ -> fl, f| -> fi)
    - Removes empty lines and trims whitespace from each line
    - Normalizes multiple spaces to single spaces
    
    Args:
        text: The raw text string to clean.
        
    Returns:
        A cleaned and normalized version of the input text.
    """
    text = text.replace("\r", "\n")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("f|", "fi")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    text = " ".join(text.split())

    return text


def load_document(
    document_text: Optional[str] = None, 
    document_file: Optional[str] = None
) -> str:
    """Load and process a document from text or PDF file.
    
    This function can load a document from either:
    - Direct text input (via `document_text` parameter)
    - A PDF file path (via `document_file` parameter)
    
    The loaded text is automatically cleaned and normalized using the
    `clean_text` function before being returned.
    
    Args:
        document_text: Optional string containing the document text directly.
            If provided, this takes precedence over `document_file`.
        document_file: Optional path to a PDF file to load. Should be a valid
            file path accessible by the application.
            
    Returns:
        A cleaned and normalized string containing the document text.
        
    Raises:
        ValueError: If neither `document_text` nor `document_file` is provided,
            if `document_text` is empty, if the PDF file contains no extractable
            text, or if there's an error reading the PDF file.
    """
    if document_text:
        text = document_text.strip()
        if not text:
            raise ValueError("No text is available for provided document.")
        return clean_text(text)
    
    if document_file:
        try:
            reader = PdfReader(document_file)
            pages_text = []

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            if not pages_text:
                raise ValueError("No text is available for provided document.")
            
            complete_text = "\n".join(pages_text)
            return clean_text(complete_text)
        except Exception as e:
            raise ValueError(f"Failed to load PDF document: {str(e)}")
            
    raise ValueError("No document text or PDF file provided.")