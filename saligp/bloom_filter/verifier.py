"""
Phase 5: Bloom Filter Module
Fast duplicate detection using Bloom filters
"""
import logging
import hashlib
from typing import Tuple, Dict, Set, Any
import pandas as pd
import numpy as np
from pathlib import Path
from config.config import BLOOM_FILTER_CONFIG, OUTPUTS_DIR

logger = logging.getLogger(__name__)


class SimpleBloomFilter:
    """Simple Bloom Filter implementation"""

    def __init__(self, size: int = 100000, num_hashes: int = 5):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
        self.inserted_count = 0

    def _hash(self, item: str, seed: int) -> int:
        """Hash function with seed"""
        h = hashlib.md5((item + str(seed)).encode())
        return int(h.hexdigest(), 16) % self.size

    def insert(self, item: str) -> None:
        """Insert item into Bloom filter"""
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            self.bit_array[idx] = True
        self.inserted_count += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in filter"""
        for i in range(self.num_hashes):
            idx = self._hash(item, i)
            if not self.bit_array[idx]:
                return False
        return True

    def false_positive_rate(self) -> float:
        """Estimate false positive rate"""
        if self.inserted_count == 0:
            return 0.0
        bits_set = sum(self.bit_array)
        return (bits_set / self.size) ** self.num_hashes


class BloomFilterVerifier:
    """
    Bloom Filter-based verification for duplicate detection
    """

    def __init__(self):
        self.bloom_filter = SimpleBloomFilter(
            size=BLOOM_FILTER_CONFIG["bloom_size"],
            num_hashes=BLOOM_FILTER_CONFIG["hash_functions"],
        )
        self.results = {}

    def verify_duplicates(
        self, hash_values: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Verify duplicates using Bloom filter"""
        logger.info("=" * 60)
        logger.info("PHASE 5: BLOOM FILTER VERIFICATION")
        logger.info("=" * 60)

        logger.info(f"\n[CONFIG]")
        logger.info(f"    Bloom size: {BLOOM_FILTER_CONFIG['bloom_size']}")
        logger.info(f"    Hash functions: {BLOOM_FILTER_CONFIG['hash_functions']}")
        logger.info(f"    Total hashes: {len(hash_values)}")

        predictions = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        lookup_times = []

        logger.info("\n[1] Verifying duplicates...")

        for i, (hash_val, label) in enumerate(zip(hash_values, labels)):
            # Record lookup time
            import time
            start = time.time()
            
            # Check if hash exists in Bloom filter
            found = self.bloom_filter.contains(hash_val)
            
            lookup_time = (time.time() - start) * 1000  # ms
            lookup_times.append(lookup_time)

            # Insert hash
            self.bloom_filter.insert(hash_val)

            # Comparison with label
            if found and label == 1:
                true_positives += 1
                predictions.append(1)
            elif found and label == 0:
                false_positives += 1
                predictions.append(1)
            elif not found and label == 1:
                false_negatives += 1
                predictions.append(0)
            else:
                predictions.append(0)

            if (i + 1) % 1000 == 0:
                logger.info(f"    Processed {i + 1}/{len(hash_values)}")

        predictions = np.array(predictions)

        # Calculate metrics
        fp_rate = self.bloom_filter.false_positive_rate()
        mean_lookup_time = np.mean(lookup_times)
        std_lookup_time = np.std(lookup_times)

        self.results = {
            "true_positives": int(true_positives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "estimated_fp_rate": float(fp_rate),
            "mean_lookup_time_ms": float(mean_lookup_time),
            "std_lookup_time_ms": float(std_lookup_time),
            "bloom_bits_set": sum(self.bloom_filter.bit_array),
        }

        logger.info(f"\n[2] Results")
        logger.info(f"    True Positives: {true_positives}")
        logger.info(f"    False Positives: {false_positives}")
        logger.info(f"    False Negatives: {false_negatives}")
        logger.info(f"    Est. FP Rate: {fp_rate:.6f}")
        logger.info(f"    Mean Lookup Time: {mean_lookup_time:.4f}ms")

        self._save_results()

        logger.info("\n" + "=" * 60)
        logger.info("✓ BLOOM FILTER VERIFICATION COMPLETE")
        logger.info("=" * 60)

        return predictions, self.results

    def _save_results(self) -> None:
        """Save Bloom filter results"""
        logger.info("\n[3] Saving Bloom filter results...")

        results_df = pd.DataFrame([self.results])
        results_path = OUTPUTS_DIR / "bloom_metrics.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"    Saved to: {results_path}")

    def get_results(self) -> Dict[str, Any]:
        """Get verification results"""
        return self.results
