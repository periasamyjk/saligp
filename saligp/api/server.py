"""
FastAPI Module
REST API for SALIGP predictions
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import io
import logging
import time
from pipeline import SALIGPClassifier
from config.config import ALL_FEATURES
from text_processing import DocumentProcessor, DocumentRecord, TextFeatureExtractor

logger = logging.getLogger(__name__)

LEGACY_TEXT_FEATURES = [
    "cosine_similarity",
    "jaccard_similarity",
    "edit_distance",
    "token_overlap",
    "word_overlap",
    "length_ratio",
    "char_ngram",
    "tfidf_cosine",
]


class PairPredictionRequest(BaseModel):
    """Request model for pair prediction"""
    pair_id: int = 1
    features: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    data: List[Dict[str, Any]]


class TextPairPredictionRequest(BaseModel):
    """Request model for raw text pair prediction"""
    pair_id: int = 1
    left_text: str
    right_text: str
    left_filename: str = "left.txt"
    right_filename: str = "right.txt"


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    pair_id: int
    is_duplicate: int
    confidence: float
    cluster_id: Optional[int] = None


class OwnershipTransferRequest(BaseModel):
    """Request model for ownership transfer"""
    pair_id: int
    from_user: str
    to_user: str


class SALIGPAPIServer:
    """
    FastAPI server for SALIGP
    """

    def __init__(self, classifier: SALIGPClassifier):
        self.classifier = classifier
        self.document_processor = DocumentProcessor()
        self.text_feature_extractor = TextFeatureExtractor()
        self.app = FastAPI(
            title="SALIGP API",
            description="Secure Active Learning with Improved Genetic Programming",
            version="1.0.0",
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes"""

        @self.app.post("/predict", response_model=PredictionResponse)
        def predict_pair(request: PairPredictionRequest):
            """
            Predict if a pair is a duplicate
            """
            try:
                # Extract features
                normalized_features = self._normalize_feature_row(request.features)
                X = self._features_to_array(normalized_features)

                # Make prediction
                pred = self.classifier.predict(X)[0]
                score = self.classifier.gp_model.predict(X)[0]
                confidence = self._decision_confidence(int(pred), float(score))

                return PredictionResponse(
                    pair_id=request.pair_id,
                    is_duplicate=int(pred),
                    confidence=confidence,
                )
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/predict-text", response_model=PredictionResponse)
        def predict_text_pair(request: TextPairPredictionRequest):
            """Predict a duplicate pair from raw text fields"""
            try:
                left = DocumentRecord(
                    document_id=f"{request.pair_id}:left",
                    filename=request.left_filename,
                    text=request.left_text,
                    size_bytes=len(request.left_text.encode("utf-8")),
                )
                right = DocumentRecord(
                    document_id=f"{request.pair_id}:right",
                    filename=request.right_filename,
                    text=request.right_text,
                    size_bytes=len(request.right_text.encode("utf-8")),
                )
                features = self.text_feature_extractor.compute_pair_features(left, right)
                X = self._features_to_array(features)
                pred, confidence = self._predict_arrays(X)
                decision_confidence = self._decision_confidence(
                    int(pred[0]),
                    float(confidence[0]),
                )
                return PredictionResponse(
                    pair_id=request.pair_id,
                    is_duplicate=int(pred[0]),
                    confidence=decision_confidence,
                )
            except Exception as e:
                logger.error(f"Text prediction error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/predict-batch")
        def predict_batch(request: BatchPredictionRequest):
            """Predict duplicate status for a batch of feature rows or text rows"""
            try:
                df = pd.DataFrame(request.data)
                result_df = self._predict_dataframe(df)
                return {"results": self._format_results(result_df)}
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/upload")
        async def upload_documents(files: List[UploadFile] = File(...)):
            """Upload documents/files and run pairwise deduplication"""
            try:
                start_time = time.perf_counter()
                if not files:
                    raise ValueError("Upload at least two documents or one CSV file.")

                filename = files[0].filename or ""
                if len(files) == 1 and filename.lower().endswith((".csv", ".tsv")):
                    content = await files[0].read()
                    df = self._read_uploaded_csv(content, filename)
                    result_df = self._predict_dataframe(df)
                else:
                    documents = []
                    for file in files:
                        content = await file.read()
                        documents.append(
                            self.document_processor.from_bytes(
                                content=content,
                                filename=file.filename or "upload.txt",
                                content_type=file.content_type,
                            )
                        )
                    result_df = self.classifier.predict_documents(documents)

                processing_time = time.perf_counter() - start_time
                if len(result_df) > 0:
                    result_df["processing_time_seconds"] = processing_time / len(result_df)

                return {
                    "total_pairs": int(len(result_df)),
                    "duplicates_found": int(result_df["saligp_prediction"].sum()),
                    "results": self._format_results(result_df),
                }
            except Exception as e:
                logger.error(f"Upload error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/upload_pair")
        def upload_pair(request: PairPredictionRequest):
            """Register a feature pair in the ownership system"""
            try:
                self.classifier.role_manager.assign_ownership(
                    request.pair_id, "default_user"
                )
                return {"pair_id": request.pair_id, "status": "uploaded"}
            except Exception as e:
                logger.error(f"Upload pair error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/ownership/{pair_id}")
        def get_ownership(pair_id: int):
            """Get ownership information for a pair"""
            try:
                ownership = self.classifier.role_manager.get_ownership(pair_id)
                return {"pair_id": pair_id, "owners": ownership}
            except Exception as e:
                logger.error(f"Ownership lookup error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/transfer_ownership")
        def transfer_ownership(request: OwnershipTransferRequest):
            """Transfer ownership of a pair"""
            try:
                success = self.classifier.role_manager.transfer_ownership(
                    request.pair_id, request.from_user, request.to_user
                )
                return {
                    "pair_id": request.pair_id,
                    "success": success,
                    "from": request.from_user,
                    "to": request.to_user,
                }
            except Exception as e:
                logger.error(f"Transfer error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/health")
        def health():
            """Health check"""
            return {"status": "ok"}

    def get_app(self) -> FastAPI:
        """Get FastAPI app"""
        return self.app

    def _read_uploaded_csv(self, content: bytes, filename: str) -> pd.DataFrame:
        delimiter = "\t" if filename.lower().endswith(".tsv") else ","
        return pd.read_csv(io.BytesIO(content), sep=delimiter)

    def _predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("No rows found for prediction.")

        if all(feature in df.columns for feature in ALL_FEATURES) or all(feature in df.columns for feature in LEGACY_TEXT_FEATURES):
            feature_rows = [
                {"pair_id": row.get("pair_id", index + 1), **self._normalize_feature_row(row.to_dict())}
                for index, row in df.iterrows()
            ]
            return self.classifier.predict_batch(pd.DataFrame(feature_rows))

        pair_rows = []
        for index, row in df.iterrows():
            left_text = self._first_present(row, ["left_text", "text_a", "document_a", "source_text"])
            right_text = self._first_present(row, ["right_text", "text_b", "document_b", "target_text"])
            if left_text is None or right_text is None:
                continue
            pair_id = int(row["pair_id"]) if "pair_id" in row and pd.notna(row["pair_id"]) else index + 1
            left_name = str(row.get("left_filename", row.get("filename_a", f"{pair_id}_left.txt")))
            right_name = str(row.get("right_filename", row.get("filename_b", f"{pair_id}_right.txt")))
            left = DocumentRecord(
                document_id=f"{pair_id}:left",
                filename=left_name,
                text=str(left_text),
                size_bytes=len(str(left_text).encode("utf-8")),
            )
            right = DocumentRecord(
                document_id=f"{pair_id}:right",
                filename=right_name,
                text=str(right_text),
                size_bytes=len(str(right_text).encode("utf-8")),
            )
            pair_rows.append({"pair_id": pair_id, **self.text_feature_extractor.compute_pair_features(left, right)})

        if not pair_rows:
            raise ValueError(
                "CSV must contain either SALIGP feature columns or text pair columns "
                "(left_text/right_text, text_a/text_b, document_a/document_b, or source_text/target_text)."
            )

        return self.classifier.predict_batch(pd.DataFrame(pair_rows))

    def _first_present(self, row: pd.Series, columns: List[str]):
        for column in columns:
            if column in row and pd.notna(row[column]):
                return row[column]
        return None

    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([[features.get(feature, 0.0) for feature in ALL_FEATURES]], dtype=np.float32)

    def _normalize_feature_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        if all(feature in row for feature in ALL_FEATURES):
            return {feature: self._as_feature_float(row.get(feature, 0.0)) for feature in ALL_FEATURES}

        legacy_values = {
            feature: self._as_feature_float(row.get(feature, 0.0))
            for feature in LEGACY_TEXT_FEATURES
        }
        semantic_overlap = np.mean(
            [legacy_values["token_overlap"], legacy_values["word_overlap"]]
        )
        lexical_overlap = np.mean(
            [legacy_values["jaccard_similarity"], legacy_values["char_ngram"]]
        )
        supplied = [legacy_values[feature] for feature in LEGACY_TEXT_FEATURES]

        return {
            "filename_similarity": legacy_values["edit_distance"],
            "content_similarity": legacy_values["cosine_similarity"],
            "metadata_similarity": float(semantic_overlap),
            "size_similarity": legacy_values["length_ratio"],
            "tfidf_similarity": legacy_values["tfidf_cosine"],
            "embedding_similarity": float(lexical_overlap),
            "sha256_match": self._as_feature_float(row.get("sha256_match", 0.0)),
            "overall_similarity": self._as_feature_float(row.get("overall_similarity", np.mean(supplied))),
        }

    def _as_feature_float(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if np.isnan(numeric) or np.isinf(numeric):
            return 0.0
        return float(max(0.0, min(1.0, numeric)))

    def _predict_arrays(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.classifier, "predict_with_confidence"):
            return self.classifier.predict_with_confidence(X)
        predictions = self.classifier.predict(X)
        scores = self.classifier.gp_model.predict(X)
        return predictions, scores

    def _decision_confidence(self, prediction: int, score: float) -> float:
        confidence = score if prediction == 1 else 1.0 - score
        return float(max(0.0, min(1.0, confidence)))

    def _format_results(self, result_df: pd.DataFrame) -> List[Dict[str, Any]]:
        confidence_col = "prediction_confidence" if "prediction_confidence" in result_df.columns else "uncertainty"
        results = []
        for index, row in result_df.iterrows():
            confidence = float(row.get(confidence_col, 0.5))
            if confidence_col == "uncertainty":
                confidence = 1.0 - confidence
            results.append(
                {
                    "id": int(row.get("pair_id", index + 1)),
                    "pair": self._pair_label(row, index),
                    "isDuplicate": bool(row.get("saligp_prediction", 0)),
                    "confidence": round(max(0.0, min(1.0, confidence)) * 100, 2),
                    "prediction_confidence": float(row.get("prediction_confidence", max(0.0, min(1.0, confidence)))),
                    "gp_score": float(row.get("gp_score", row.get("prediction_confidence", confidence))),
                    "uncertainty": float(row.get("uncertainty", max(0.0, min(1.0, 1.0 - confidence)))),
                    "cluster": f"Cluster {int(row.get('cluster_id', row.get('cluster', 0)))}",
                    "input_size_kb": self._optional_float(row.get("input_size_kb")),
                    "input_file": self._optional_string(row.get("input_file", row.get("left_filename"))),
                    "duplicate_file": self._optional_string(row.get("duplicate_file", row.get("right_filename"))),
                    "actual_size_bytes": self._optional_int(row.get("actual_size_bytes")),
                    "content_profile": self._optional_string(row.get("content_profile")),
                    "sha256_prefix": self._optional_string(row.get("sha256_prefix")),
                    "processing_time_seconds": self._optional_float(row.get("processing_time_seconds")),
                    "features": {
                        feature: float(row[feature])
                        for feature in ALL_FEATURES
                        if feature in row and pd.notna(row[feature])
                    },
                }
            )
        return results

    def _pair_label(self, row: pd.Series, index: int) -> str:
        left = row.get("left_filename")
        right = row.get("right_filename")
        if pd.notna(left) and pd.notna(right):
            return f"{left} <-> {right}"
        return f"Pair_{index + 1:03d}"

    def _optional_string(self, value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        return str(value)

    def _optional_float(self, value: Any) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _optional_int(self, value: Any) -> Optional[int]:
        parsed = self._optional_float(value)
        return int(parsed) if parsed is not None else None
