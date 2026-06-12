"""
Phase 1: Data Validation Module
Validates dataset integrity and consistency
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from config.config import (
    FEATURE_RANGES,
    ALL_FEATURES,
    TARGET_COLUMN,
    OUTPUTS_DIR,
    DIFFICULTY_MAPPING,
)
from data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates SALIGP dataset for integrity and consistency
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.validation_report = {}
        self.is_valid = True

    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA VALIDATION")
        logger.info("=" * 60)

        self.validate_missing_values()
        self.validate_duplicate_rows()
        self.validate_labels()
        self.validate_difficulty_codes()
        self.validate_feature_ranges()
        self.validate_train_validation_test_consistency()
        self.validate_pair_id_consistency()
        self.validate_saligp_features_structure()

        self._generate_report()

        return self.is_valid

    def validate_missing_values(self) -> None:
        """Check for missing values"""
        logger.info("\n[1] Checking for missing values...")

        datasets = {
            "train": self.data_loader.train_df,
            "validation": self.data_loader.validation_df,
            "test": self.data_loader.test_df,
            "saligp_features": self.data_loader.saligp_features_df,
            "al_seed": self.data_loader.al_seed_df,
            "al_pool": self.data_loader.al_pool_df,
            "gp_training": self.data_loader.gp_training_df,
        }

        for name, df in datasets.items():
            missing = df.isnull().sum().sum()
            if missing > 0:
                logger.error(f"    {name}: {missing} missing values found!")
                self.is_valid = False
            else:
                logger.info(f"    {name}: OK")

            self.validation_report[f"{name}_missing_values"] = missing

    def validate_duplicate_rows(self) -> None:
        """Check for duplicate rows"""
        logger.info("\n[2] Checking for duplicate rows...")

        datasets = {
            "train": self.data_loader.train_df,
            "validation": self.data_loader.validation_df,
            "test": self.data_loader.test_df,
            "gp_training": self.data_loader.gp_training_df,
        }

        for name, df in datasets.items():
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"    {name}: {duplicates} duplicate rows found")
            else:
                logger.info(f"    {name}: OK (no duplicates)")

            self.validation_report[f"{name}_duplicate_rows"] = duplicates

    def validate_labels(self) -> None:
        """Validate label distribution"""
        logger.info("\n[3] Validating label distribution...")

        for name, df in [
            ("train", self.data_loader.train_df),
            ("validation", self.data_loader.validation_df),
            ("test", self.data_loader.test_df),
            ("saligp_features", self.data_loader.saligp_features_df),
        ]:
            if TARGET_COLUMN not in df.columns:
                logger.error(f"    {name}: Missing '{TARGET_COLUMN}' column!")
                self.is_valid = False
                continue

            unique_labels = df[TARGET_COLUMN].unique()
            label_counts = df[TARGET_COLUMN].value_counts()

            if not set(unique_labels).issubset({0, 1}):
                logger.error(
                    f"    {name}: Invalid labels found: {unique_labels}"
                )
                self.is_valid = False
            else:
                logger.info(
                    f"    {name}: Labels = {dict(label_counts)}"
                )

            self.validation_report[f"{name}_label_distribution"] = (
                label_counts.to_dict()
            )

    def validate_difficulty_codes(self) -> None:
        """Validate difficulty codes"""
        logger.info("\n[4] Validating difficulty codes...")

        df = self.data_loader.saligp_features_df

        if "difficulty" not in df.columns:
            logger.error("    Missing 'difficulty' column!")
            self.is_valid = False
            return

        valid_difficulties = set(DIFFICULTY_MAPPING.keys())
        invalid = df[~df["difficulty"].isin(valid_difficulties)]["difficulty"].unique()

        if len(invalid) > 0:
            logger.warning(f"    Invalid difficulty values: {invalid}")
        else:
            logger.info("    OK: All difficulty values are valid")

        diff_counts = df["difficulty"].value_counts()
        logger.info(f"    Distribution: {dict(diff_counts)}")
        self.validation_report["difficulty_distribution"] = diff_counts.to_dict()

    def validate_feature_ranges(self) -> None:
        """Validate feature value ranges"""
        logger.info("\n[5] Validating feature ranges...")

        df = self.data_loader.saligp_features_df

        for feature in ALL_FEATURES:
            if feature not in df.columns:
                logger.warning(f"    Missing feature: {feature}")
                continue

            min_val = df[feature].min()
            max_val = df[feature].max()
            expected_min, expected_max = FEATURE_RANGES[feature]

            if min_val < expected_min or max_val > expected_max:
                logger.warning(
                    f"    {feature}: Out of range [{min_val}, {max_val}] "
                    f"(expected [{expected_min}, {expected_max}])"
                )
            else:
                logger.info(f"    {feature}: OK [{min_val:.4f}, {max_val:.4f}]")

            self.validation_report[f"{feature}_range"] = (
                float(min_val),
                float(max_val),
            )

    def validate_train_validation_test_consistency(self) -> None:
        """Check consistency between train/val/test splits"""
        logger.info("\n[6] Checking train/validation/test consistency...")

        train_size = len(self.data_loader.train_df)
        val_size = len(self.data_loader.validation_df)
        test_size = len(self.data_loader.test_df)
        total_size = train_size + val_size + test_size

        logger.info(
            f"    Train: {train_size} ({train_size/total_size*100:.1f}%)"
        )
        logger.info(
            f"    Validation: {val_size} ({val_size/total_size*100:.1f}%)"
        )
        logger.info(f"    Test: {test_size} ({test_size/total_size*100:.1f}%)")

        train_cols = set(self.data_loader.train_df.columns)
        val_cols = set(self.data_loader.validation_df.columns)
        test_cols = set(self.data_loader.test_df.columns)

        if train_cols != val_cols or train_cols != test_cols:
            logger.error("    Column mismatch between datasets!")
            self.is_valid = False
        else:
            logger.info("    Column consistency: OK")

        self.validation_report["train_size"] = train_size
        self.validation_report["validation_size"] = val_size
        self.validation_report["test_size"] = test_size

    def validate_pair_id_consistency(self) -> None:
        """Check pair_id uniqueness"""
        logger.info("\n[7] Checking pair_id consistency...")

        df = self.data_loader.saligp_features_df

        if "pair_id" not in df.columns:
            logger.warning("    No pair_id column found")
            return

        n_total = len(df)
        n_unique = df["pair_id"].nunique()

        if n_unique == n_total:
            logger.info(f"    OK: All {n_unique} pair_ids are unique")
        else:
            logger.warning(
                f"    {n_total - n_unique} duplicate pair_ids found"
            )

        self.validation_report["pair_id_uniqueness"] = n_unique == n_total

    def validate_saligp_features_structure(self) -> None:
        """Validate SALIGP features dataframe structure"""
        logger.info("\n[8] Validating SALIGP features structure...")

        df = self.data_loader.saligp_features_df
        required_cols = [
            "pair_id",
            "label",
            "geometric_cluster_id",
            "uncertainty_score",
            "difficulty_code",
        ]

        for col in required_cols:
            if col not in df.columns:
                logger.error(f"    Missing required column: {col}")
                self.is_valid = False
            else:
                logger.info(f"    {col}: OK")

        self.validation_report["required_columns_present"] = all(
            col in df.columns for col in required_cols
        )

    def _generate_report(self) -> None:
        """Generate HTML validation report"""
        logger.info("\n[REPORT] Generating validation report...")

        html_content = self._create_html_report()

        report_path = OUTPUTS_DIR / "dataset_validation_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"    Report saved to: {report_path}")
        self.validation_report["report_path"] = str(report_path)

        if self.is_valid:
            logger.info("\n" + "=" * 60)
            logger.info("✓ DATA VALIDATION PASSED")
            logger.info("=" * 60)
        else:
            logger.error("\n" + "=" * 60)
            logger.error("✗ DATA VALIDATION FAILED - Review report")
            logger.error("=" * 60)

    def _create_html_report(self) -> str:
        """Create HTML report content"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SALIGP Dataset Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #0066cc; margin-top: 20px; }}
        .section {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .status-pass {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>SALIGP Dataset Validation Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    
    <div class="section">
        <h2>Validation Status</h2>
        <p><strong>Overall Status:</strong> 
            <span class="{'status-pass' if self.is_valid else 'status-fail'}">
                {'PASSED ✓' if self.is_valid else 'FAILED ✗'}
            </span>
        </p>
    </div>
    
    <div class="section">
        <h2>Dataset Sizes</h2>
        <table>
            <tr><th>Dataset</th><th>Size</th></tr>
            <tr><td>Training</td><td>{self.validation_report.get('train_size', 0)}</td></tr>
            <tr><td>Validation</td><td>{self.validation_report.get('validation_size', 0)}</td></tr>
            <tr><td>Test</td><td>{self.validation_report.get('test_size', 0)}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Label Distribution</h2>
        <table>
            <tr><th>Dataset</th><th>Label 0</th><th>Label 1</th></tr>
            <tr>
                <td>Train</td>
                <td>{self.validation_report.get('train_label_distribution', {}).get(0, 0)}</td>
                <td>{self.validation_report.get('train_label_distribution', {}).get(1, 0)}</td>
            </tr>
            <tr>
                <td>Validation</td>
                <td>{self.validation_report.get('validation_label_distribution', {}).get(0, 0)}</td>
                <td>{self.validation_report.get('validation_label_distribution', {}).get(1, 0)}</td>
            </tr>
            <tr>
                <td>Test</td>
                <td>{self.validation_report.get('test_label_distribution', {}).get(0, 0)}</td>
                <td>{self.validation_report.get('test_label_distribution', {}).get(1, 0)}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Difficulty Distribution</h2>
        <table>
            <tr><th>Difficulty</th><th>Count</th></tr>
"""

        diff_dist = self.validation_report.get("difficulty_distribution", {})
        for diff, count in diff_dist.items():
            html += f"            <tr><td>{diff}</td><td>{count}</td></tr>\n"

        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Feature Ranges</h2>
        <table>
            <tr><th>Feature</th><th>Min</th><th>Max</th><th>Expected Range</th></tr>
"""

        for feature in ALL_FEATURES:
            min_val, max_val = self.validation_report.get(
                f"{feature}_range", (0, 1)
            )
            expected = FEATURE_RANGES[feature]
            html += f"""            <tr>
                <td>{feature}</td>
                <td>{min_val:.4f}</td>
                <td>{max_val:.4f}</td>
                <td>[{expected[0]}, {expected[1]}]</td>
            </tr>\n"""

        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>Validation Checks</h2>
        <ul>
"""

        checks = [
            ("Missing Values", "train_missing_values" in self.validation_report),
            ("Duplicate Rows", "train_duplicate_rows" in self.validation_report),
            ("Label Consistency", "train_label_distribution" in self.validation_report),
            ("Feature Ranges", "filename_similarity_range" in self.validation_report),
            ("Pair ID Uniqueness", self.validation_report.get("pair_id_uniqueness", False)),
            ("Required Columns", self.validation_report.get("required_columns_present", False)),
        ]

        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            color = "green" if check_result else "red"
            html += f'<li style="color: {color};"><strong>{status} {check_name}</strong></li>\n'

        html += """
        </ul>
    </div>
</body>
</html>
"""
        return html
