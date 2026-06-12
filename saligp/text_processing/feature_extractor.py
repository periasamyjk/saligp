"""
Raw text to SALIGP feature-vector conversion.

The output schema intentionally matches config.ALL_FEATURES so existing SALIGP
models can deduplicate raw documents without changing the trained model shape.
"""
from __future__ import annotations

import hashlib
import math
import re
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


class TextFeatureExtractor:
    """Compute normalized SALIGP features for document/text pairs."""

    def documents_to_pairs(self, documents: Iterable[DocumentRecord]) -> pd.DataFrame:
        docs = list(documents)
        rows = []
        for pair_id, (left, right) in enumerate(combinations(docs, 2), start=1):
            rows.append(
                {
                    "pair_id": pair_id,
                    "left_document_id": left.document_id,
                    "right_document_id": right.document_id,
                    "left_filename": left.filename,
                    "right_filename": right.filename,
                    "input_file": left.filename,
                    "duplicate_file": right.filename,
                    "input_size_kb": round((left.size_bytes or len(left.text.encode("utf-8"))) / 1024, 3),
                    "actual_size_bytes": left.size_bytes or len(left.text.encode("utf-8")),
                    "content_profile": self._content_profile(left),
                    "sha256_prefix": self._sha256(left.text)[:16],
                    **self.compute_pair_features(left, right),
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

    def _char_ngram_similarity(self, left_text: str, right_text: str, n: int) -> float:
        left_ngrams = self._char_ngrams(left_text, n)
        right_ngrams = self._char_ngrams(right_text, n)
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
