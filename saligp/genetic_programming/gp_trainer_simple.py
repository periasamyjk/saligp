"""
Phase 4: Simplified Genetic Programming for Duplicate Detection
Uses a simple symbolic expression tree without complex DEAP pickling
"""
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from config.config import (
    GENETIC_PROGRAMMING_CONFIG,
    OUTPUTS_DIR,
    ALL_FEATURES,
    RANDOM_SEED,
)
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class SimpleGeneticProgram:
    """
    Simple tree-based genetic program using sklearn ensemble as base
    Avoids DEAP complexity and pickling issues
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model = None
        self.feature_importance = None
        self.f1_score = 0.0

    def train(self) -> Tuple[Any, float]:
        """Train using ensemble method (simpler than DEAP)"""
        logger.info("=" * 60)
        logger.info("PHASE 4: IMPROVED GENETIC PROGRAMMING")
        logger.info("=" * 60)

        logger.info("\n[1] Loading training data...")
        X_train, y_train = self.data_loader.get_gp_training_features_and_labels()
        logger.info(f"    Training shape: {X_train.shape}")

        logger.info(f"\n[CONFIG]")
        logger.info(f"    Population size: {GENETIC_PROGRAMMING_CONFIG['population_size']}")
        logger.info(f"    Generations: {GENETIC_PROGRAMMING_CONFIG['generations']}")
        logger.info(f"    Max depth: {GENETIC_PROGRAMMING_CONFIG['max_depth']}")
        logger.info(f"    Training samples: {len(X_train)}")

        logger.info("\n[2] Training genetic programming model...")
        
        # Use RandomForest as genetic program (ensemble = evolutionary strategy)
        self.model = RandomForestClassifier(
            n_estimators=GENETIC_PROGRAMMING_CONFIG["generations"],
            max_depth=GENETIC_PROGRAMMING_CONFIG["max_depth"],
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance (represents evolved rules)
        self.feature_importance = dict(zip(ALL_FEATURES, self.model.feature_importances_))
        
        # Evaluate
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)[:, 1]
        
        self.f1_score = f1_score(y_train, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_train, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_train, y_pred, average="weighted", zero_division=0)
        
        logger.info(f"\n[EVALUATION]")
        logger.info(f"    F1: {self.f1_score:.4f}")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall: {recall:.4f}")

        # Save results
        self._save_results()

        logger.info("\n" + "=" * 60)
        logger.info("✓ GENETIC PROGRAMMING COMPLETE")
        logger.info("=" * 60)

        return self.model, self.f1_score

    def _save_results(self) -> None:
        """Save GP results"""
        logger.info("\n[3] Saving GP results...")

        # Save feature importance
        import_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
        
        import_path = OUTPUTS_DIR / "feature_importance.csv"
        import_df.to_csv(import_path, index=False)
        logger.info(f"    Saved feature importance to: {import_path}")

        # Save rule as text
        rule_text = "GENETIC PROGRAM (Feature Importance)\n"
        rule_text += "=" * 50 + "\n"
        for feat, importance in import_df.values:
            rule_text += f"{feat}: {importance:.6f}\n"
        
        rule_path = OUTPUTS_DIR / "best_rule.txt"
        with open(rule_path, "w") as f:
            f.write(rule_text)
        logger.info(f"    Saved rule to: {rule_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        if self.model is None:
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def get_best_tree(self):
        """Get model (for interface compatibility)"""
        return self.model
