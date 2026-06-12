"""Text and document ingestion for SALIGP."""
from .document_processor import DocumentRecord, DocumentProcessor, UnsupportedDocumentError
from .feature_extractor import TextFeatureExtractor

__all__ = [
    "DocumentRecord",
    "DocumentProcessor",
    "TextFeatureExtractor",
    "UnsupportedDocumentError",
]
