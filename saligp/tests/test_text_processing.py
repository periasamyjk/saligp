import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SALIGP_ROOT = Path(__file__).resolve().parents[1]
if str(SALIGP_ROOT) not in sys.path:
    sys.path.insert(0, str(SALIGP_ROOT))

from config.config import ALL_FEATURES
from pipeline import SALIGPClassifier
from text_processing import DocumentProcessor, DocumentRecord, TextFeatureExtractor


class FakeGPModel:
    def predict(self, X):
        return X[:, ALL_FEATURES.index("overall_similarity")]


def make_docx(text):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>"
                f"{text}"
                "</w:t></w:r></w:p></w:body></w:document>"
            ),
        )
    return content.getvalue()


def test_document_processor_extracts_text_and_docx():
    processor = DocumentProcessor()

    txt = processor.from_bytes(b"Hello duplicate world", "sample.txt")
    docx = processor.from_bytes(make_docx("Hello duplicate world"), "sample.docx")

    assert txt.text == "Hello duplicate world"
    assert docx.text == "Hello duplicate world"


def test_feature_extractor_outputs_normalized_model_features():
    extractor = TextFeatureExtractor()
    left = DocumentRecord("1", "a.txt", "same text for duplicate detection", 33)
    right = DocumentRecord("2", "b.txt", "same text for duplicate detection", 33)

    features = extractor.compute_pair_features(left, right)

    assert list(features) == ALL_FEATURES
    assert all(0.0 <= value <= 1.0 for value in features.values())
    assert features["sha256_match"] == 1.0
    assert features["overall_similarity"] > 0.8


def test_classifier_predict_documents_runs_pairwise_deduplication():
    classifier = SALIGPClassifier(
        gp_model=FakeGPModel(),
        uncertainty_scores={},
        cluster_assignments=np.array([]),
    )
    documents = [
        DocumentRecord("1", "one.txt", "duplicate text block", 20),
        DocumentRecord("2", "two.txt", "duplicate text block", 20),
        DocumentRecord("3", "three.txt", "unrelated invoice content", 25),
    ]

    result = classifier.predict_documents(documents)

    assert len(result) == 3
    assert "saligp_prediction" in result.columns
    matching_pair = result[
        (result["left_filename"] == "one.txt") & (result["right_filename"] == "two.txt")
    ].iloc[0]
    assert matching_pair["saligp_prediction"] == 1
