"""
Data Loader Module
Handles all data loading and preprocessing
"""
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from config.config import (
    TRAIN_CSV,
    VALIDATION_CSV,
    TEST_CSV,
    SALIGP_FEATURES_CSV,
    AL_SEED_CSV,
    AL_POOL_CSV,
    GP_TRAINING_CSV,
    ALL_FEATURES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads all required datasets for SALIGP framework
    """

    def __init__(self):
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.saligp_features_df = None
        self.al_seed_df = None
        self.al_pool_df = None
        self.gp_training_df = None

    def load_all_data(self) -> None:
        """Load all datasets"""
        logger.info("Loading all datasets...")

        self.train_df = pd.read_csv(TRAIN_CSV)
        logger.info(f"Loaded train.csv: {self.train_df.shape}")

        self.validation_df = pd.read_csv(VALIDATION_CSV)
        logger.info(f"Loaded validation.csv: {self.validation_df.shape}")

        self.test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Loaded test.csv: {self.test_df.shape}")

        self.saligp_features_df = pd.read_csv(SALIGP_FEATURES_CSV)
        logger.info(f"Loaded saligp_features.csv: {self.saligp_features_df.shape}")

        self.al_seed_df = pd.read_csv(AL_SEED_CSV)
        logger.info(f"Loaded active_learning_seed.csv: {self.al_seed_df.shape}")

        self.al_pool_df = pd.read_csv(AL_POOL_CSV)
        logger.info(f"Loaded active_learning_pool.csv: {self.al_pool_df.shape}")

        self.gp_training_df = pd.read_csv(GP_TRAINING_CSV)
        logger.info(f"Loaded gp_training.csv: {self.gp_training_df.shape}")

    def get_train_features_and_labels(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get training features and labels"""
        X = self.train_df[ALL_FEATURES].values.astype(np.float32)
        y = self.train_df[TARGET_COLUMN].values.astype(np.int32)
        return X, y

    def get_validation_features_and_labels(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation features and labels"""
        X = self.validation_df[ALL_FEATURES].values.astype(np.float32)
        y = self.validation_df[TARGET_COLUMN].values.astype(np.int32)
        return X, y

    def get_test_features_and_labels(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get test features and labels"""
        X = self.test_df[ALL_FEATURES].values.astype(np.float32)
        y = self.test_df[TARGET_COLUMN].values.astype(np.int32)
        return X, y

    def get_gp_training_features_and_labels(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get GP training features and labels"""
        X = self.gp_training_df[ALL_FEATURES].values.astype(np.float32)
        y = self.gp_training_df[TARGET_COLUMN].values.astype(np.int32)
        return X, y

    def get_saligp_features_for_clustering(self) -> np.ndarray:
        """Get features for clustering (geometric analysis)"""
        clustering_features = [
            "filename_similarity",
            "content_similarity",
            "metadata_similarity",
            "size_similarity",
            "tfidf_similarity",
            "embedding_similarity",
            "overall_similarity",
        ]
        return self.saligp_features_df[clustering_features].values.astype(
            np.float32
        )

    def get_pair_ids(self) -> np.ndarray:
        """Get pair IDs from saligp_features"""
        return self.saligp_features_df["pair_id"].values

    def get_saligp_full_dataframe(self) -> pd.DataFrame:
        """Get full SALIGP features dataframe"""
        return self.saligp_features_df.copy()


class DataLoaderSingleton:
    """Singleton pattern for DataLoader"""

    _instance = None

    @classmethod
    def get_instance(cls) -> DataLoader:
        if cls._instance is None:
            cls._instance = DataLoader()
            cls._instance.load_all_data()
        return cls._instance
