"""
Phase 2: Geometric Analysis Layer (Clustering)
Implements geometric clustering using KMeans and DBSCAN
"""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
from config.config import (
    CLUSTERING_CONFIG,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class GeometricAnalyzer:
    """
    Performs geometric analysis and clustering on SALIGP features
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.X_features = None
        self.X_scaled = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_model = None
        self.scaler = StandardScaler()
        self.cluster_assignments = None
        self.results = {}

    def analyze(self) -> np.ndarray:
        """Run full geometric analysis and return cluster assignments"""
        logger.info("=" * 60)
        logger.info("PHASE 2: GEOMETRIC ANALYSIS LAYER")
        logger.info("=" * 60)

        self._load_features()
        self._scale_features()
        self._perform_kmeans()
        self._perform_dbscan()
        self._perform_pca()
        self._perform_silhouette_analysis()
        self._generate_cluster_statistics()
        self._save_results()

        return self.cluster_assignments

    def _load_features(self) -> None:
        """Load features for clustering"""
        logger.info("\n[1] Loading features for clustering...")
        self.X_features = self.data_loader.get_saligp_features_for_clustering()
        logger.info(f"    Shape: {self.X_features.shape}")

    def _scale_features(self) -> None:
        """Scale features to [0, 1]"""
        logger.info("\n[2] Scaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X_features)
        logger.info(f"    Scaling complete. Mean: {self.X_scaled.mean():.4f}, "
                   f"Std: {self.X_scaled.std():.4f}")

    def _perform_kmeans(self) -> None:
        """Perform KMeans clustering"""
        logger.info("\n[3] Performing KMeans clustering...")

        n_clusters = CLUSTERING_CONFIG["n_clusters_kmeans"]
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_SEED,
            n_init=CLUSTERING_CONFIG["kmeans_n_init"],
        )

        kmeans_labels = self.kmeans_model.fit_predict(self.X_scaled)
        inertia = self.kmeans_model.inertia_

        logger.info(f"    n_clusters: {n_clusters}")
        logger.info(f"    Inertia: {inertia:.4f}")
        logger.info(f"    Cluster distribution: {np.bincount(kmeans_labels)}")

        self.cluster_assignments = kmeans_labels
        self.results["kmeans"] = {
            "n_clusters": n_clusters,
            "inertia": float(inertia),
            "cluster_distribution": np.bincount(kmeans_labels).tolist(),
        }

    def _perform_dbscan(self) -> None:
        """Perform DBSCAN clustering"""
        logger.info("\n[4] Performing DBSCAN clustering...")

        eps = CLUSTERING_CONFIG["dbscan_eps"]
        min_samples = CLUSTERING_CONFIG["dbscan_min_samples"]

        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = self.dbscan_model.fit_predict(self.X_scaled)

        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        logger.info(f"    eps: {eps}")
        logger.info(f"    min_samples: {min_samples}")
        logger.info(f"    Clusters found: {n_clusters}")
        logger.info(f"    Noise points: {n_noise}")

        self.results["dbscan"] = {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    def _perform_pca(self) -> None:
        """Perform PCA for visualization"""
        logger.info("\n[5] Performing PCA for visualization...")

        n_components = CLUSTERING_CONFIG["pca_components"]
        self.pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = self.pca_model.fit_transform(self.X_scaled)

        explained_var = self.pca_model.explained_variance_ratio_
        total_var = sum(explained_var)

        logger.info(f"    Components: {n_components}")
        logger.info(f"    Explained variance: {explained_var}")
        logger.info(f"    Total variance explained: {total_var:.4f}")

        self.results["pca"] = {
            "n_components": n_components,
            "explained_variance_ratio": explained_var.tolist(),
            "total_variance_explained": float(total_var),
        }

        # Save PCA results for visualization
        self.results["X_pca"] = X_pca

    def _perform_silhouette_analysis(self) -> None:
        """Perform silhouette analysis"""
        logger.info("\n[6] Performing silhouette analysis...")

        silhouette_avg = silhouette_score(self.X_scaled, self.cluster_assignments)
        logger.info(f"    Silhouette score: {silhouette_avg:.4f}")

        self.results["silhouette"] = {
            "score": float(silhouette_avg),
        }

    def _generate_cluster_statistics(self) -> None:
        """Generate cluster statistics"""
        logger.info("\n[7] Generating cluster statistics...")

        unique_clusters = np.unique(self.cluster_assignments)
        stats = {}

        for cluster_id in unique_clusters:
            mask = self.cluster_assignments == cluster_id
            cluster_data = self.X_features[mask]

            stats[int(cluster_id)] = {
                "size": int(np.sum(mask)),
                "mean_features": cluster_data.mean(axis=0).tolist(),
                "std_features": cluster_data.std(axis=0).tolist(),
            }

        self.results["cluster_statistics"] = stats
        logger.info(f"    Cluster statistics computed for {len(stats)} clusters")

    def _save_results(self) -> None:
        """Save clustering results"""
        logger.info("\n[8] Saving clustering results...")

        # Update SALIGP features with cluster assignments
        df = self.data_loader.get_saligp_full_dataframe()
        df["geometric_cluster_id"] = self.cluster_assignments

        output_path = OUTPUTS_DIR / "cluster_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"    Saved cluster results to: {output_path}")

        # Save geometric assignments
        assignments_df = pd.DataFrame(
            {
                "pair_id": self.data_loader.get_pair_ids(),
                "geometric_cluster_id": self.cluster_assignments,
            }
        )
        assignments_path = OUTPUTS_DIR / "geometric_assignments.csv"
        assignments_df.to_csv(assignments_path, index=False)
        logger.info(f"    Saved assignments to: {assignments_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ GEOMETRIC ANALYSIS COMPLETE")
        logger.info("=" * 60)

    def get_cluster_assignments(self) -> np.ndarray:
        """Get cluster assignments"""
        return self.cluster_assignments

    def get_pca_features(self) -> np.ndarray:
        """Get PCA-transformed features"""
        return self.results.get("X_pca")
