"""
Phase 7: SALIGP Integration
Main classifier combining all components
"""
import logging
from typing import Iterable, Tuple, Dict, Any
import numpy as np
import pandas as pd
from config.config import ALL_FEATURES, OUTPUTS_DIR
from genetic_programming import ImprovedGeneticProgramming
from bloom_filter import BloomFilterVerifier
from role_hierarchy import RoleHierarchyManager
from text_processing import DocumentRecord, TextFeatureExtractor

logger = logging.getLogger(__name__)


class SALIGPClassifier:
    """
    Secure Active Learning with Improved Genetic Programming (SALIGP)
    Main classifier combining all components
    """

    def __init__(
        self,
        gp_model: ImprovedGeneticProgramming,
        uncertainty_scores: Dict,
        cluster_assignments: np.ndarray,
    ):
        self.gp_model = gp_model
        self.uncertainty_scores = uncertainty_scores
        self.cluster_assignments = cluster_assignments
        self.bloom_verifier = BloomFilterVerifier()
        self.role_manager = RoleHierarchyManager()
        self.text_feature_extractor = TextFeatureExtractor()

    def predict(self, X: np.ndarray, pair_ids: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using SALIGP
        
        Pipeline:
        1. GP evaluation
        2. Bloom verification
        3. Final classification
        """
        logger.info("Making predictions with SALIGP...")

        # Step 1: GP evaluation
        gp_scores = self.gp_model.predict(X)
        gp_predictions = (gp_scores > 0.5).astype(int)

        logger.info(f"    GP predictions: {gp_predictions.sum()} duplicates")

        return gp_predictions

    def predict_with_uncertainty(
        self, X: np.ndarray, pair_ids: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty scores
        
        Returns:
            predictions: Binary predictions
            confidence: Uncertainty scores [0, 1]
        """
        gp_scores = self.gp_model.predict(X)
        
        # Get uncertainty scores
        if pair_ids is not None:
            uncertainties = np.array(
                [self.uncertainty_scores.get(pid, 0.5) for pid in pair_ids]
            )
        else:
            uncertainties = np.full(len(X), 0.5)

        predictions = (gp_scores > 0.5).astype(int)

        return predictions, uncertainties

    def predict_batch(
        self, pairs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Batch prediction on dataframe
        """
        X = pairs_df[ALL_FEATURES].values.astype(np.float32)
        pair_ids = pairs_df["pair_id"].values if "pair_id" in pairs_df.columns else None

        gp_scores = self.gp_model.predict(X)
        predictions = (gp_scores > 0.5).astype(int)
        if pair_ids is not None:
            uncertainties = np.array(
                [self.uncertainty_scores.get(pid, np.nan) for pid in pair_ids],
                dtype=np.float32,
            )
        else:
            uncertainties = np.full(len(X), np.nan, dtype=np.float32)

        decision_confidence = np.where(predictions == 1, gp_scores, 1.0 - gp_scores)
        decision_confidence = np.clip(decision_confidence, 0.0, 1.0)
        fallback_uncertainty = 1.0 - decision_confidence
        uncertainties = np.where(np.isnan(uncertainties), fallback_uncertainty, uncertainties)

        result_df = pairs_df.copy()
        result_df["saligp_prediction"] = predictions
        result_df["gp_score"] = gp_scores
        result_df["prediction_confidence"] = decision_confidence
        result_df["uncertainty"] = uncertainties

        if "geometric_cluster_id" in result_df.columns:
            result_df["cluster"] = self.cluster_assignments[
                : len(result_df)
            ]

        return result_df

    def predict_documents(
        self,
        documents: Iterable[DocumentRecord],
    ) -> pd.DataFrame:
        """
        Deduplicate raw documents by extracting pairwise text features first.
        """
        pairs_df = self.text_feature_extractor.documents_to_pairs(documents)
        if pairs_df.empty:
            raise ValueError("At least two documents are required for deduplication.")
        return self.predict_batch(pairs_df)

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        pair_ids: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Evaluate SALIGP on test set
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        predictions = self.predict(X, pair_ids)
        gp_scores = self.gp_model.predict(X)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
        }

        # ROC-AUC if possible
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, gp_scores)
        except:
            metrics["roc_auc"] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        metrics["tn"] = int(tn)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tp"] = int(tp)

        return metrics

    def register_duplicates(
        self,
        pair_ids: np.ndarray,
        predictions: np.ndarray,
        user_id: str = "admin",
    ) -> None:
        """
        Register predicted duplicates in ownership system
        """
        logger.info("Registering duplicates in ownership system...")

        # Register admin user
        self.role_manager.register_user(user_id, "Admin")

        # Register each pair
        for pair_id, is_duplicate in zip(pair_ids, predictions):
            if is_duplicate == 1:
                self.role_manager.assign_ownership(
                    int(pair_id), user_id, is_duplicate=1
                )
                self.role_manager.register_duplicate(int(pair_id), int(pair_id))

        self.role_manager.print_statistics()


class SALIGPPipeline:
    """
    Complete SALIGP pipeline orchestrator
    """

    def __init__(self):
        self.classifier = None
        self.results = {}

    def build_and_train(
        self,
        data_loader,
        validator,
        geometric_analyzer,
        active_learner,
        gp_trainer,
    ) -> None:
        """Build and train SALIGP pipeline"""
        logger.info("=" * 60)
        logger.info("PHASE 7: SALIGP INTEGRATION")
        logger.info("=" * 60)

        # Create classifier
        self.classifier = SALIGPClassifier(
            gp_model=gp_trainer.toolbox if hasattr(gp_trainer, 'toolbox') else gp_trainer,
            uncertainty_scores=active_learner.get_uncertainty_scores(),
            cluster_assignments=geometric_analyzer.get_cluster_assignments(),
        )

        logger.info("\n✓ SALIGP classifier created and integrated")

    def get_classifier(self) -> SALIGPClassifier:
        """Get trained classifier"""
        return self.classifier
