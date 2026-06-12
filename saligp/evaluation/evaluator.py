"""
Phase 8: Evaluation Module
Comprehensive evaluation of SALIGP framework
"""
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from pathlib import Path
from config.config import OUTPUTS_DIR, DIFFICULTY_MAPPING, DIFFICULTY_REVERSE_MAPPING
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class SALIGPEvaluator:
    """
    Comprehensive evaluation of SALIGP predictions
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.results = {}

    def evaluate_full_pipeline(
        self,
        test_predictions: np.ndarray,
        test_scores: np.ndarray,
        test_pair_ids: np.ndarray = None,
    ) -> Dict:
        """Comprehensive evaluation"""
        logger.info("=" * 60)
        logger.info("PHASE 8: EVALUATION")
        logger.info("=" * 60)

        # Get test labels
        X_test, y_test = self.data_loader.get_test_features_and_labels()
        test_df = self.data_loader.test_df.copy()

        # Overall metrics
        logger.info("\n[1] Overall Metrics")
        overall_metrics = self._compute_metrics(y_test, test_predictions, test_scores)
        
        for metric, value in overall_metrics.items():
            logger.info(f"    {metric}: {value:.4f}")

        self.results["overall"] = overall_metrics

        # Difficulty-wise evaluation
        logger.info("\n[2] Difficulty-wise Evaluation")
        difficulty_results = self._evaluate_by_difficulty(
            test_df, test_predictions, test_scores
        )

        for difficulty, metrics in difficulty_results.items():
            logger.info(f"    {difficulty}:")
            for metric, value in metrics.items():
                logger.info(f"      {metric}: {value:.4f}")

        self.results["by_difficulty"] = difficulty_results

        # Cluster-wise evaluation (if available)
        if "geometric_cluster_id" in test_df.columns:
            logger.info("\n[3] Cluster-wise Evaluation")
            cluster_results = self._evaluate_by_cluster(
                test_df, test_predictions, test_scores
            )

            for cluster, metrics in cluster_results.items():
                logger.info(f"    Cluster {cluster}:")
                for metric, value in metrics.items():
                    logger.info(f"      {metric}: {value:.4f}")

            self.results["by_cluster"] = cluster_results

        # Save results
        self._save_evaluation_results()

        logger.info("\n" + "=" * 60)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 60)

        return self.results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray = None,
    ) -> Dict[str, float]:
        """Compute standard metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
                metrics["pr_auc"] = average_precision_score(y_true, y_score)
            except:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["tp"] = float(tp)
        metrics["fp"] = float(fp)
        metrics["tn"] = float(tn)
        metrics["fn"] = float(fn)
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return metrics

    def _evaluate_by_difficulty(
        self, test_df: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray
    ) -> Dict:
        """Evaluate by difficulty level"""
        results = {}

        if "difficulty" not in test_df.columns:
            return results

        for difficulty in test_df["difficulty"].unique():
            mask = test_df["difficulty"] == difficulty
            
            y_true = test_df.loc[mask, "label"].values
            y_pred = predictions[mask]
            y_score = scores[mask] if scores is not None else None

            if len(y_true) > 0:
                metrics = self._compute_metrics(y_true, y_pred, y_score)
                results[difficulty] = metrics

        return results

    def _evaluate_by_cluster(
        self, test_df: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray
    ) -> Dict:
        """Evaluate by cluster"""
        results = {}

        if "geometric_cluster_id" not in test_df.columns:
            return results

        for cluster_id in test_df["geometric_cluster_id"].unique():
            mask = test_df["geometric_cluster_id"] == cluster_id
            
            y_true = test_df.loc[mask, "label"].values
            y_pred = predictions[mask]
            y_score = scores[mask] if scores is not None else None

            if len(y_true) > 0:
                metrics = self._compute_metrics(y_true, y_pred, y_score)
                results[int(cluster_id)] = metrics

        return results

    def _save_evaluation_results(self) -> None:
        """Save evaluation results to CSV"""
        logger.info("\n[SAVE] Saving evaluation results...")

        # Overall results
        overall_df = pd.DataFrame([self.results["overall"]])
        overall_path = OUTPUTS_DIR / "evaluation_overall.csv"
        overall_df.to_csv(overall_path, index=False)
        logger.info(f"    Saved overall results to: {overall_path}")

        # By difficulty
        if "by_difficulty" in self.results:
            difficulty_df = pd.DataFrame(
                [
                    {"difficulty": k, **v}
                    for k, v in self.results["by_difficulty"].items()
                ]
            )
            difficulty_path = OUTPUTS_DIR / "evaluation_by_difficulty.csv"
            difficulty_df.to_csv(difficulty_path, index=False)
            logger.info(f"    Saved difficulty results to: {difficulty_path}")

        # By cluster
        if "by_cluster" in self.results:
            cluster_df = pd.DataFrame(
                [
                    {"cluster": k, **v}
                    for k, v in self.results["by_cluster"].items()
                ]
            )
            cluster_path = OUTPUTS_DIR / "evaluation_by_cluster.csv"
            cluster_df.to_csv(cluster_path, index=False)
            logger.info(f"    Saved cluster results to: {cluster_path}")

    def get_results(self) -> Dict:
        """Get evaluation results"""
        return self.results


class BaselineComparison:
    """
    Compare SALIGP with baseline methods
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.results = {}

    def run_comparisons(
        self,
        saligp_predictions: np.ndarray,
        saligp_scores: np.ndarray,
        al_predictions: np.ndarray = None,
        gp_predictions: np.ndarray = None,
        rf_predictions: np.ndarray = None,
        lr_predictions: np.ndarray = None,
    ) -> Dict:
        """Compare SALIGP with baselines"""
        logger.info("=" * 60)
        logger.info("BASELINE COMPARISON")
        logger.info("=" * 60)

        X_test, y_test = self.data_loader.get_test_features_and_labels()

        comparison_data = []

        # SALIGP
        saligp_metrics = self._evaluate_model(
            y_test, saligp_predictions, saligp_scores, "SALIGP"
        )
        comparison_data.append(saligp_metrics)

        # AL only
        if al_predictions is not None:
            al_metrics = self._evaluate_model(
                y_test, al_predictions, None, "AL Only"
            )
            comparison_data.append(al_metrics)

        # GP only
        if gp_predictions is not None:
            gp_metrics = self._evaluate_model(
                y_test, gp_predictions, None, "GP Only"
            )
            comparison_data.append(gp_metrics)

        # Random Forest
        if rf_predictions is not None:
            rf_metrics = self._evaluate_model(
                y_test, rf_predictions, None, "RandomForest"
            )
            comparison_data.append(rf_metrics)

        # Logistic Regression
        if lr_predictions is not None:
            lr_metrics = self._evaluate_model(
                y_test, lr_predictions, None, "LogisticRegression"
            )
            comparison_data.append(lr_metrics)

        # Save comparison
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = OUTPUTS_DIR / "baseline_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison to: {comparison_path}")

        return {row["model"]: row for row in comparison_data}

    def _evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        model_name: str,
    ) -> Dict:
        """Evaluate a single model"""
        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
                metrics["pr_auc"] = average_precision_score(y_true, y_score)
            except:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        logger.info(f"\n{model_name}:")
        for key, value in metrics.items():
            if key != "model":
                logger.info(f"    {key}: {value:.4f}")

        return metrics
