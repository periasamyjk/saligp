"""
Phase 3: Active Learning Layer
Implements uncertainty sampling for label selection
"""
import logging
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from config.config import (
    ACTIVE_LEARNING_CONFIG,
    OUTPUTS_DIR,
    ALL_FEATURES,
    RANDOM_SEED,
)
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Implements Active Learning with uncertainty sampling
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.al_seed_df = data_loader.al_seed_df.copy()
        self.al_pool_df = data_loader.al_pool_df.copy()
        self.val_df = data_loader.validation_df.copy()
        
        self.labeled_set = self.al_seed_df.copy()
        self.unlabeled_pool = self.al_pool_df.copy()
        
        self.rf_model = None
        self.lr_model = None
        self.history = []
        self.uncertainty_scores = {}

    def run_active_learning(self) -> Tuple[np.ndarray, List[Dict]]:
        """Run active learning loop"""
        logger.info("=" * 60)
        logger.info("PHASE 3: ACTIVE LEARNING LAYER")
        logger.info("=" * 60)

        n_iterations = ACTIVE_LEARNING_CONFIG["n_iterations"]
        logger.info(f"\n[CONFIG] Running {n_iterations} AL iterations...")

        for iteration in range(n_iterations):
            logger.info(f"\n[ITERATION {iteration + 1}/{n_iterations}]")
            
            # Train models
            self._train_models()
            
            # Evaluate
            metrics = self._evaluate_on_validation()
            self.history.append({"iteration": iteration + 1, **metrics})
            
            # Calculate uncertainty for pool
            uncertainties = self._calculate_uncertainty()
            
            # Select most uncertain samples
            if len(self.unlabeled_pool) > 0:
                selected_indices = self._select_samples(uncertainties)
                self._add_to_labeled_set(selected_indices)
            
            logger.info(f"    Labeled set size: {len(self.labeled_set)}")
            logger.info(f"    Unlabeled pool size: {len(self.unlabeled_pool)}")

        self._calculate_final_uncertainty_scores()
        self._save_results()
        
        return self.labeled_set[ALL_FEATURES].values, self.history

    def _train_models(self) -> None:
        """Train RF and LR models on labeled set"""
        X_train = self.labeled_set[ALL_FEATURES].values.astype(np.float32)
        y_train = self.labeled_set["label"].values.astype(np.int32)
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=RANDOM_SEED,
        )
        self.rf_model.fit(X_train, y_train)
        
        # Logistic Regression
        self.lr_model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
        )
        self.lr_model.fit(X_train, y_train)

    def _evaluate_on_validation(self) -> Dict:
        """Evaluate on validation set"""
        X_val = self.val_df[ALL_FEATURES].values.astype(np.float32)
        y_val = self.val_df["label"].values.astype(np.int32)
        
        # RF predictions
        rf_pred = self.rf_model.predict(X_val)
        rf_proba = self.rf_model.predict_proba(X_val)[:, 1]
        
        # LR predictions
        lr_pred = self.lr_model.predict(X_val)
        lr_proba = self.lr_model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "rf_accuracy": accuracy_score(y_val, rf_pred),
            "rf_f1": f1_score(y_val, rf_pred, average="weighted"),
            "rf_roc_auc": roc_auc_score(y_val, rf_proba),
            "lr_accuracy": accuracy_score(y_val, lr_pred),
            "lr_f1": f1_score(y_val, lr_pred, average="weighted"),
            "lr_roc_auc": roc_auc_score(y_val, lr_proba),
        }
        
        logger.info(f"    RF Acc: {metrics['rf_accuracy']:.4f}, F1: {metrics['rf_f1']:.4f}, ROC: {metrics['rf_roc_auc']:.4f}")
        logger.info(f"    LR Acc: {metrics['lr_accuracy']:.4f}, F1: {metrics['lr_f1']:.4f}, ROC: {metrics['lr_roc_auc']:.4f}")
        
        return metrics

    def _calculate_uncertainty(self) -> np.ndarray:
        """Calculate uncertainty scores for unlabeled pool"""
        if len(self.unlabeled_pool) == 0:
            return np.array([])
        
        X_pool = self.unlabeled_pool[ALL_FEATURES].values.astype(np.float32)
        
        # Entropy-based uncertainty
        rf_proba = self.rf_model.predict_proba(X_pool)
        entropy = -np.sum(rf_proba * np.log(rf_proba + 1e-10), axis=1)
        
        return entropy

    def _select_samples(self, uncertainties: np.ndarray) -> np.ndarray:
        """Select most uncertain samples"""
        n_select = min(
            ACTIVE_LEARNING_CONFIG["n_samples_per_iteration"],
            len(self.unlabeled_pool),
        )
        
        # Select top-k most uncertain
        top_indices = np.argsort(uncertainties)[-n_select:]
        
        return top_indices

    def _add_to_labeled_set(self, indices: np.ndarray) -> None:
        """Add selected samples to labeled set"""
        selected = self.unlabeled_pool.iloc[indices].copy()
        self.labeled_set = pd.concat([self.labeled_set, selected], ignore_index=True)
        self.unlabeled_pool = self.unlabeled_pool.drop(
            self.unlabeled_pool.index[indices]
        ).reset_index(drop=True)

    def _calculate_final_uncertainty_scores(self) -> None:
        """Calculate uncertainty scores for all samples"""
        logger.info("\n[FINAL] Calculating uncertainty scores for all samples...")
        
        df = self.data_loader.get_saligp_full_dataframe()
        X_all = df[ALL_FEATURES].values.astype(np.float32)
        
        rf_proba = self.rf_model.predict_proba(X_all)
        entropy = -np.sum(rf_proba * np.log(rf_proba + 1e-10), axis=1)
        
        # Normalize to [0, 1]
        uncertainty_scores = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        
        self.uncertainty_scores = dict(zip(df["pair_id"], uncertainty_scores))
        
        logger.info(f"    Uncertainty scores calculated: {len(self.uncertainty_scores)}")
        logger.info(f"    Mean uncertainty: {np.mean(uncertainty_scores):.4f}")
        logger.info(f"    Std uncertainty: {np.std(uncertainty_scores):.4f}")

    def _save_results(self) -> None:
        """Save AL results"""
        logger.info("\n[SAVE] Saving Active Learning results...")
        
        # Save learning history
        history_df = pd.DataFrame(self.history)
        history_path = OUTPUTS_DIR / "active_learning_results.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"    Saved AL results to: {history_path}")
        
        # Save uncertainty scores
        df = self.data_loader.get_saligp_full_dataframe()
        df["uncertainty_score"] = df["pair_id"].map(self.uncertainty_scores)
        uncertainty_path = OUTPUTS_DIR / "updated_uncertainty_scores.csv"
        df.to_csv(uncertainty_path, index=False)
        logger.info(f"    Saved uncertainty scores to: {uncertainty_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ACTIVE LEARNING COMPLETE")
        logger.info("=" * 60)

    def get_uncertainty_scores(self) -> Dict:
        """Get uncertainty scores"""
        return self.uncertainty_scores

    def get_learned_model(self) -> RandomForestClassifier:
        """Get trained RF model"""
        return self.rf_model
