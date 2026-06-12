"""
SALIGP Framework Configuration
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "saligp" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Data files
TRAIN_CSV = DATA_DIR / "pairs" / "train.csv"
VALIDATION_CSV = DATA_DIR / "pairs" / "validation.csv"
TEST_CSV = DATA_DIR / "pairs" / "test.csv"
SALIGP_FEATURES_CSV = DATA_DIR / "pairs" / "saligp_features.csv"
AL_SEED_CSV = DATA_DIR / "pairs" / "active_learning_seed.csv"
AL_POOL_CSV = DATA_DIR / "pairs" / "active_learning_pool.csv"
GP_TRAINING_CSV = DATA_DIR / "pairs" / "gp_training.csv"

# Feature columns
SIMILARITY_FEATURES = [
    "filename_similarity",
    "content_similarity",
    "metadata_similarity",
    "size_similarity",
    "tfidf_similarity",
    "embedding_similarity",
]

ALL_FEATURES = SIMILARITY_FEATURES + ["sha256_match", "overall_similarity"]

TARGET_COLUMN = "label"
CLUSTER_COLUMN = "geometric_cluster_id"
UNCERTAINTY_COLUMN = "uncertainty_score"
DIFFICULTY_COLUMN = "difficulty_code"

# Difficulty mapping
DIFFICULTY_MAPPING = {
    "Easy": 0,
    "Medium": 1,
    "Hard": 2,
    "Non-Duplicate": 4,
}

DIFFICULTY_REVERSE_MAPPING = {v: k for k, v in DIFFICULTY_MAPPING.items()}

# Similarity feature ranges
FEATURE_RANGES = {
    "filename_similarity": (0.0, 1.0),
    "content_similarity": (0.0, 1.0),
    "metadata_similarity": (0.0, 1.0),
    "size_similarity": (0.0, 1.0),
    "tfidf_similarity": (0.0, 1.0),
    "embedding_similarity": (0.0, 1.0),
    "sha256_match": (0, 1),
    "overall_similarity": (0.0, 1.0),
}

# ============================================
# PHASE 2: GEOMETRIC CLUSTERING CONFIG
# ============================================

CLUSTERING_CONFIG = {
    "n_clusters_kmeans": 4,
    "kmeans_random_state": 42,
    "kmeans_n_init": 10,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "pca_components": 2,
}

# ============================================
# PHASE 3: ACTIVE LEARNING CONFIG
# ============================================

ACTIVE_LEARNING_CONFIG = {
    "initial_labeled_size": 100,
    "n_iterations": 10,
    "n_samples_per_iteration": 50,
    "random_state": 42,
    "test_size": 0.2,
    "uncertainty_methods": ["entropy", "margin", "least_confident"],
}

# ============================================
# PHASE 4: GENETIC PROGRAMMING CONFIG
# ============================================

GENETIC_PROGRAMMING_CONFIG = {
    "population_size": 100,
    "generations": 50,
    "max_depth": 8,
    "min_depth": 2,
    "tournament_size": 3,
    "cx_probability": 0.7,
    "mut_probability": 0.3,
    "elite_size": 2,
    "fitness_sample_size": 1200,
    "random_state": 42,
    "timeout_seconds": 3600,
}

# ============================================
# PHASE 5: BLOOM FILTER CONFIG
# ============================================

BLOOM_FILTER_CONFIG = {
    "bloom_size": 100000,
    "hash_functions": 5,
}

# ============================================
# PHASE 6: ROLE HIERARCHY CONFIG
# ============================================

ROLE_HIERARCHY_CONFIG = {
    "db_path": str(OUTPUTS_DIR / "saligp_ownership.db"),
    "roles": ["Admin", "Manager", "Employee"],
}

# ============================================
# EVALUATION CONFIG
# ============================================

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    "average_method": "weighted",
}

# ============================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================

RANDOM_SEED = 42

# ============================================
# LOGGING CONFIG
# ============================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }
    },
}
