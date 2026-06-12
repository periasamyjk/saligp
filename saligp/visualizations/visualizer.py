"""
Visualization Module
Generate plots and visualizations for SALIGP results
"""
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from config.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class SALIGPVisualizer:
    """
    Generates visualizations for SALIGP framework
    """

    def __init__(self):
        self.output_dir = OUTPUTS_DIR
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_pca_clusters(
        self, X_pca: np.ndarray, labels: np.ndarray, cluster_ids: np.ndarray
    ) -> None:
        """Plot PCA with cluster assignments"""
        logger.info("\n[VIZ] Generating PCA cluster plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ensure matching lengths
        n_samples = len(X_pca)
        labels_plot = labels[:n_samples] if len(labels) >= n_samples else np.pad(labels, (0, n_samples - len(labels)))
        clusters_plot = cluster_ids[:n_samples] if len(cluster_ids) >= n_samples else np.pad(cluster_ids, (0, n_samples - len(cluster_ids)))

        # Plot by label
        scatter1 = axes[0].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels_plot.astype(int),
            cmap="viridis",
            s=50,
            alpha=0.6,
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title("PCA Clustering by Label")
        plt.colorbar(scatter1, ax=axes[0], label="Label")

        # Plot by cluster
        scatter2 = axes[1].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters_plot.astype(int),
            cmap="tab10",
            s=50,
            alpha=0.6,
        )
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("PCA Clustering by Cluster ID")
        plt.colorbar(scatter2, ax=axes[1], label="Cluster ID")

        plt.tight_layout()
        output_path = self.output_dir / "pca_clusters.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"    Saved to: {output_path}")

    def plot_cluster_distribution(self, cluster_ids: np.ndarray) -> None:
        """Plot cluster distribution"""
        logger.info("\n[VIZ] Generating cluster distribution plot...")

        unique, counts = np.unique(cluster_ids, return_counts=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(unique, counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Count")
        ax.set_title("Cluster Distribution")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "cluster_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"    Saved to: {output_path}")

    def plot_learning_curve(self, history: List[Dict]) -> None:
        """Plot active learning curve"""
        logger.info("\n[VIZ] Generating learning curve...")

        df = pd.DataFrame(history)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RF Metrics
        axes[0, 0].plot(df["iteration"], df["rf_accuracy"], marker="o", label="Accuracy")
        axes[0, 0].plot(df["iteration"], df["rf_f1"], marker="s", label="F1")
        axes[0, 0].plot(df["iteration"], df["rf_roc_auc"], marker="^", label="ROC-AUC")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Random Forest Performance")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # LR Metrics
        axes[0, 1].plot(df["iteration"], df["lr_accuracy"], marker="o", label="Accuracy")
        axes[0, 1].plot(df["iteration"], df["lr_f1"], marker="s", label="F1")
        axes[0, 1].plot(df["iteration"], df["lr_roc_auc"], marker="^", label="ROC-AUC")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Logistic Regression Performance")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Comparison
        axes[1, 0].plot(
            df["iteration"], df["rf_accuracy"], marker="o", label="RF Accuracy"
        )
        axes[1, 0].plot(
            df["iteration"], df["lr_accuracy"], marker="s", label="LR Accuracy"
        )
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Model Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # F1 Comparison
        axes[1, 1].plot(df["iteration"], df["rf_f1"], marker="o", label="RF F1")
        axes[1, 1].plot(df["iteration"], df["lr_f1"], marker="s", label="LR F1")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_title("F1 Score Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "learning_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"    Saved to: {output_path}")

    def plot_evaluation_metrics(self, metrics: Dict) -> None:
        """Plot evaluation metrics"""
        logger.info("\n[VIZ] Generating evaluation metrics plot...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Overall metrics
        metric_names = ["accuracy", "precision", "recall", "f1"]
        metric_values = [metrics["overall"].get(m, 0) for m in metric_names]

        axes[0, 0].bar(metric_names, metric_values, color="steelblue", edgecolor="black")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Overall Metrics")
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis="y", alpha=0.3)

        # Difficulty-wise F1
        if "by_difficulty" in metrics:
            difficulties = list(metrics["by_difficulty"].keys())
            f1_scores = [metrics["by_difficulty"][d].get("f1", 0) for d in difficulties]

            axes[0, 1].bar(difficulties, f1_scores, color="coral", edgecolor="black")
            axes[0, 1].set_ylabel("F1 Score")
            axes[0, 1].set_title("F1 Score by Difficulty")
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(axis="y", alpha=0.3)

        # Confusion Matrix
        if "overall" in metrics:
            m = metrics["overall"]
            cm = np.array([[m.get("tn", 0), m.get("fp", 0)],
                          [m.get("fn", 0), m.get("tp", 0)]])
            
            im = axes[1, 0].imshow(cm, cmap="Blues", aspect="auto")
            axes[1, 0].set_xlabel("Predicted")
            axes[1, 0].set_ylabel("Actual")
            axes[1, 0].set_title("Confusion Matrix")
            axes[1, 0].set_xticks([0, 1])
            axes[1, 0].set_yticks([0, 1])
            axes[1, 0].set_xticklabels(["Neg", "Pos"])
            axes[1, 0].set_yticklabels(["Neg", "Pos"])
            
            for i in range(2):
                for j in range(2):
                    axes[1, 0].text(j, i, str(int(cm[i, j])),
                                   ha="center", va="center", color="white")

        # ROC-AUC and PR-AUC
        if "overall" in metrics:
            m = metrics["overall"]
            auc_scores = [
                m.get("roc_auc", 0),
                m.get("pr_auc", 0),
            ]
            auc_names = ["ROC-AUC", "PR-AUC"]

            axes[1, 1].bar(auc_names, auc_scores, color="lightgreen", edgecolor="black")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_title("AUC Scores")
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "evaluation_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"    Saved to: {output_path}")

    def plot_baseline_comparison(self, comparison: Dict) -> None:
        """Plot baseline comparison"""
        logger.info("\n[VIZ] Generating baseline comparison plots...")

        models = list(comparison.keys())
        accuracies = [comparison[m].get("accuracy", 0) for m in models]
        f1_scores = [comparison[m].get("f1", 0) for m in models]
        precisions = [comparison[m].get("precision", 0) for m in models]
        recalls = [comparison[m].get("recall", 0) for m in models]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy
        axes[0, 0].bar(models, accuracies, color="steelblue", edgecolor="black")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Accuracy Comparison")
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis="y", alpha=0.3)
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # F1
        axes[0, 1].bar(models, f1_scores, color="coral", edgecolor="black")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("F1 Score Comparison")
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis="y", alpha=0.3)
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Precision
        axes[1, 0].bar(models, precisions, color="lightgreen", edgecolor="black")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision Comparison")
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis="y", alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Recall
        axes[1, 1].bar(models, recalls, color="gold", edgecolor="black")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].set_title("Recall Comparison")
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis="y", alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        output_path = self.output_dir / "baseline_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"    Saved to: {output_path}")
