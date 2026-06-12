import sys
from pathlib import Path

import numpy as np
import pandas as pd

SALIGP_ROOT = Path(__file__).resolve().parents[1]
if str(SALIGP_ROOT) not in sys.path:
    sys.path.insert(0, str(SALIGP_ROOT))

from api.server import SALIGPAPIServer
from config.config import ALL_FEATURES
from pipeline import SALIGPClassifier


class FakeGPModel:
    def predict(self, X):
        return X[:, ALL_FEATURES.index("overall_similarity")]


def make_client():
    classifier = SALIGPClassifier(
        gp_model=FakeGPModel(),
        uncertainty_scores={},
        cluster_assignments=np.array([]),
    )
    return SALIGPAPIServer(classifier)


def test_upload_documents_returns_pairwise_predictions():
    server = make_client()

    documents = [
        server.document_processor.from_bytes(b"same duplicate text", "one.txt", "text/plain"),
        server.document_processor.from_bytes(b"same duplicate text", "two.txt", "text/plain"),
    ]
    result = server.classifier.predict_documents(documents)
    payload = {"results": server._format_results(result)}

    assert payload["results"][0]["isDuplicate"] is True


def test_upload_csv_text_pairs_returns_predictions():
    server = make_client()
    df = pd.DataFrame([{"pair_id": 7, "left_text": "alpha beta", "right_text": "alpha beta"}])

    result = server._predict_dataframe(df)
    payload = {"results": server._format_results(result)}

    assert payload["results"][0]["id"] == 7
    assert payload["results"][0]["isDuplicate"] is True


def test_legacy_feature_columns_are_normalized_for_prediction():
    server = make_client()
    df = pd.DataFrame(
        [
            {
                "pair_id": 9,
                "cosine_similarity": 1.0,
                "jaccard_similarity": 1.0,
                "edit_distance": 1.0,
                "token_overlap": 1.0,
                "word_overlap": 1.0,
                "length_ratio": 1.0,
                "char_ngram": 1.0,
                "tfidf_cosine": 1.0,
            }
        ]
    )

    result = server._predict_dataframe(df)
    payload = {"results": server._format_results(result)}

    assert payload["results"][0]["id"] == 9
    assert payload["results"][0]["isDuplicate"] is True
