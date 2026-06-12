"""
Raw text to SALIGP feature-vector conversion.

The output schema intentionally matches config.ALL_FEATURES so existing SALIGP
models can deduplicate raw documents without changing the trained model shape.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.config import ALL_FEATURES
from .document_processor import DocumentRecord


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class _PreparedDocument:
    record: DocumentRecord
    normalized_text: str
    tokens: List[str]
    token_counts: Counter
    char_ngrams: set[str]
    sha256: str
    size: int
    extension: str
    content_profile: str


class TextFeatureExtractor:
    """Compute normalized SALIGP features for document/text pairs."""

    def documents_to_pairs(self, documents: Iterable[DocumentRecord]) -> pd.DataFrame:
        docs = [self._prepare_document(document) for document in documents]
        tfidf_matrix = self._tfidf_matrix([doc.normalized_text for doc in docs])
        rows = []
        for pair_id, (left_index, right_index) in enumerate(combinations(range(len(docs)), 2), start=1):
            left = docs[left_index]
            right = docs[right_index]
            features = self._compute_prepared_pair_features(
                left,
                right,
                self._tfidf_pair_similarity(tfidf_matrix, left_index, right_index),
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "left_document_id": left.record.document_id,
                    "right_document_id": right.record.document_id,
                    "left_filename": left.record.filename,
                    "right_filename": right.record.filename,
                    "input_file": left.record.filename,
                    "duplicate_file": right.record.filename,
                    "input_size_kb": round(left.size / 1024, 3),
                    "actual_size_bytes": left.size,
                    "content_profile": left.content_profile,
                    "sha256_prefix": left.sha256[:16],
                    **features,
                }
            )
        return pd.DataFrame(rows)

    def compute_pair_features(
        self,
        left: DocumentRecord,
        right: DocumentRecord,
    ) -> Dict[str, float]:
        left_text = self._normalize_text(left.text)
        right_text = self._normalize_text(right.text)
        left_tokens = self._tokens(left_text)
        right_tokens = self._tokens(right_text)

        content_similarity = self._cosine_counter(left_tokens, right_tokens)
        tfidf_similarity = self._tfidf_cosine(left_text, right_text)
        embedding_similarity = self._char_ngram_similarity(left_text, right_text, n=3)
        filename_similarity = self._sequence_similarity(left.filename, right.filename)
        metadata_similarity = self._metadata_similarity(left, right)
        size_similarity = self._size_similarity(left.size_bytes or len(left.text), right.size_bytes or len(right.text))
        sha256_match = 1.0 if self._sha256(left.text) == self._sha256(right.text) and left.text else 0.0

        components = [
            filename_similarity,
            content_similarity,
            metadata_similarity,
            size_similarity,
            tfidf_similarity,
            embedding_similarity,
            sha256_match,
        ]
        overall_similarity = float(np.mean(components))

        features = {
            "filename_similarity": filename_similarity,
            "content_similarity": content_similarity,
            "metadata_similarity": metadata_similarity,
            "size_similarity": size_similarity,
            "tfidf_similarity": tfidf_similarity,
            "embedding_similarity": embedding_similarity,
            "sha256_match": sha256_match,
            "overall_similarity": overall_similarity,
        }
        return {name: self._clamp(features[name]) for name in ALL_FEATURES}

    def _prepare_document(self, document: DocumentRecord) -> _PreparedDocument:
        normalized_text = self._normalize_text(document.text)
        tokens = self._tokens(normalized_text)
        size = document.size_bytes or len(document.text.encode("utf-8"))
        extension = self._extension(document.filename)
        return _PreparedDocument(
            record=document,
            normalized_text=normalized_text,
            tokens=tokens,
            token_counts=Counter(tokens),
            char_ngrams=self._char_ngrams(normalized_text, n=3),
            sha256=self._sha256(document.text),
            size=size,
            extension=extension,
            content_profile=self._content_profile(document),
        )

    def _compute_prepared_pair_features(
        self,
        left: _PreparedDocument,
        right: _PreparedDocument,
        tfidf_similarity: float,
    ) -> Dict[str, float]:
        content_similarity = self._cosine_counter_counts(left.token_counts, right.token_counts)
        embedding_similarity = self._char_ngram_set_similarity(left.char_ngrams, right.char_ngrams)
        filename_similarity = self._sequence_similarity(left.record.filename, right.record.filename)
        metadata_similarity = self._prepared_metadata_similarity(left, right)
        size_similarity = self._size_similarity(left.size, right.size)
        sha256_match = 1.0 if left.sha256 == right.sha256 and left.record.text else 0.0

        components = [
            filename_similarity,
            content_similarity,
            metadata_similarity,
            size_similarity,
            tfidf_similarity,
            embedding_similarity,
            sha256_match,
        ]
        overall_similarity = float(np.mean(components))

        features = {
            "filename_similarity": filename_similarity,
            "content_similarity": content_similarity,
            "metadata_similarity": metadata_similarity,
            "size_similarity": size_similarity,
            "tfidf_similarity": tfidf_similarity,
            "embedding_similarity": embedding_similarity,
            "sha256_match": sha256_match,
            "overall_similarity": overall_similarity,
        }
        return {name: self._clamp(features[name]) for name in ALL_FEATURES}

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _tokens(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def _sequence_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left.lower(), right.lower()).ratio()

    def _cosine_counter(self, left_tokens: List[str], right_tokens: List[str]) -> float:
        if not left_tokens or not right_tokens:
            return 0.0
        left_counts = Counter(left_tokens)
        right_counts = Counter(right_tokens)
        return self._cosine_counter_counts(left_counts, right_counts)

    def _cosine_counter_counts(self, left_counts: Counter, right_counts: Counter) -> float:
        if not left_counts or not right_counts:
            return 0.0
        vocabulary = set(left_counts) | set(right_counts)
        dot = sum(left_counts[token] * right_counts[token] for token in vocabulary)
        left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
        right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _tfidf_cosine(self, left_text: str, right_text: str) -> float:
        if not left_text or not right_text:
            return 0.0
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform([left_text, right_text])
            return float(cosine_similarity(matrix[0], matrix[1])[0, 0])
        except ValueError:
            return 0.0

    def _tfidf_matrix(self, documents: List[str]):
        if not documents or not any(documents):
            return None
        try:
            return TfidfVectorizer(stop_words="english").fit_transform(documents)
        except ValueError:
            return None

    def _tfidf_pair_similarity(self, matrix, left_index: int, right_index: int) -> float:
        if matrix is None:
            return 0.0
        return float(matrix[left_index].multiply(matrix[right_index]).sum())

    def _char_ngram_similarity(self, left_text: str, right_text: str, n: int) -> float:
        left_ngrams = self._char_ngrams(left_text, n)
        right_ngrams = self._char_ngrams(right_text, n)
        return self._char_ngram_set_similarity(left_ngrams, right_ngrams)

    def _char_ngram_set_similarity(self, left_ngrams: set[str], right_ngrams: set[str]) -> float:
        if not left_ngrams or not right_ngrams:
            return 0.0
        intersection = len(left_ngrams & right_ngrams)
        union = len(left_ngrams | right_ngrams)
        return intersection / union if union else 0.0

    def _char_ngrams(self, text: str, n: int) -> set[str]:
        compact = re.sub(r"\s+", " ", text)
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[index : index + n] for index in range(len(compact) - n + 1)}

    def _metadata_similarity(self, left: DocumentRecord, right: DocumentRecord) -> float:
        matches = 0.0
        total = 2.0
        if left.content_type and right.content_type and left.content_type == right.content_type:
            matches += 1.0
        if self._extension(left.filename) == self._extension(right.filename):
            matches += 1.0
        return matches / total

    def _prepared_metadata_similarity(self, left: _PreparedDocument, right: _PreparedDocument) -> float:
        matches = 0.0
        total = 2.0
        if (
            left.record.content_type
            and right.record.content_type
            and left.record.content_type == right.record.content_type
        ):
            matches += 1.0
        if left.extension == right.extension:
            matches += 1.0
        return matches / total

    def _size_similarity(self, left_size: int, right_size: int) -> float:
        larger = max(left_size, right_size)
        if larger <= 0:
            return 1.0
        return min(left_size, right_size) / larger

    def _extension(self, filename: str) -> str:
        return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    def _content_profile(self, document: DocumentRecord) -> str:
        extension = self._extension(document.filename)
        if document.content_type:
            return f"{extension or 'uploaded'} document ({document.content_type})"
        return f"{extension or 'uploaded'} document"

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    def _clamp(self, value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))
