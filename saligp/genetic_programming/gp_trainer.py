"""
Phase 4: Improved Genetic Programming (IGP) Layer
Implements DEAP-based genetic programming for duplicate detection
"""
import logging
import operator
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from functools import partial
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import f1_score, precision_score, recall_score
from config.config import (
    GENETIC_PROGRAMMING_CONFIG,
    OUTPUTS_DIR,
    ALL_FEATURES,
    RANDOM_SEED,
)
from data_loader import DataLoader

logger = logging.getLogger(__name__)


def protected_divide(a: float, b: float) -> float:
    """Protected division to avoid division by zero"""
    return a / b if abs(b) > 1e-12 else 1.0


def protected_logistic(x: float) -> float:
    """Convert a tree output into a bounded duplicate score."""
    value = float(x)
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + np.exp(-max(-60.0, min(60.0, value))))


def greater_than(a: float, b: float) -> float:
    """Greater than comparison"""
    return 1.0 if a > b else 0.0


def less_than(a: float, b: float) -> float:
    """Less than comparison"""
    return 1.0 if a < b else 0.0


def logical_and(a: float, b: float) -> float:
    """Logical AND"""
    return 1.0 if (a > 0.5 and b > 0.5) else 0.0


def logical_or(a: float, b: float) -> float:
    """Logical OR"""
    return 1.0 if (a > 0.5 or b > 0.5) else 0.0


class ImprovedGeneticProgramming:
    """
    Implements the paper-aligned IGP tree for duplicate detection.

    Evidence leaves are the SALIGP similarity features; internal nodes evolve
    arithmetic and logical constraints; F1-score is the fitness function.
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.toolbox = None
        self.pset = None
        self.best_individual = None
        self.logbook = None
        self.evolution_stats = []
        self.best_fitness = 0.0
        self.compile_context = self._compile_context()

    def train(self) -> Tuple["ImprovedGeneticProgramming", float]:
        """Train GP model"""
        logger.info("=" * 60)
        logger.info("PHASE 4: IMPROVED GENETIC PROGRAMMING")
        logger.info("=" * 60)

        self._setup_deap()
        X_train, y_train = self._get_training_data()

        logger.info(f"\n[CONFIG]")
        logger.info(f"    Population size: {GENETIC_PROGRAMMING_CONFIG['population_size']}")
        logger.info(f"    Generations: {GENETIC_PROGRAMMING_CONFIG['generations']}")
        logger.info(f"    Max depth: {GENETIC_PROGRAMMING_CONFIG['max_depth']}")
        logger.info(f"    Training samples: {len(X_train)}")

        self._run_evolution(X_train, y_train)
        self.best_fitness = self._evaluate_best(X_train, y_train)
        self._save_results()

        return self, self.best_fitness

    def _compile_context(self) -> Dict[str, Any]:
        return {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": protected_divide,
            "gt": greater_than,
            "lt": less_than,
            "and_": logical_and,
            "or_": logical_or,
            "min": min,
            "max": max,
            "avg": lambda a, b: (a + b) / 2.0,
        }

    def _setup_deap(self) -> None:
        """Setup DEAP framework"""
        logger.info("\n[1] Setting up DEAP framework...")
        np.random.seed(RANDOM_SEED)

        # Define primitives
        self.pset = gp.PrimitiveSet("MAIN", len(ALL_FEATURES))
        
        # Terminal symbols
        self.pset.addPrimitive(np.add, 2, name="add")
        self.pset.addPrimitive(np.subtract, 2, name="sub")
        self.pset.addPrimitive(np.multiply, 2, name="mul")
        self.pset.addPrimitive(protected_divide, 2, name="div")
        self.pset.addPrimitive(greater_than, 2, name="gt")
        self.pset.addPrimitive(less_than, 2, name="lt")
        self.pset.addPrimitive(logical_and, 2, name="and_")
        self.pset.addPrimitive(logical_or, 2, name="or_")
        self.pset.addPrimitive(min, 2, name="min")
        self.pset.addPrimitive(max, 2, name="max")
        self.pset.addPrimitive(lambda a, b: (a + b) / 2.0, 2, name="avg")

        # Ephemeral constants
        self.pset.addEphemeralConstant("const", partial(np.random.uniform, 0.0, 1.0))

        # Rename arguments to feature names
        for i, feat in enumerate(ALL_FEATURES):
            self.pset.renameArguments(**{f"ARG{i}": feat})

        # Create fitness and individual
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=GENETIC_PROGRAMMING_CONFIG["min_depth"],
            max_=GENETIC_PROGRAMMING_CONFIG["max_depth"],
        )
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        logger.info(f"    Primitive set created with {sum(len(v) for v in self.pset.primitives.values())} primitives")

    def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data"""
        logger.info("\n[2] Loading training data...")
        X_train, y_train = self.data_loader.get_gp_training_features_and_labels()
        logger.info(f"    Training shape: {X_train.shape}")
        sample_size = GENETIC_PROGRAMMING_CONFIG.get("fitness_sample_size")
        if sample_size and len(X_train) > sample_size:
            rng = np.random.default_rng(RANDOM_SEED)
            sampled_indices = []
            for label in np.unique(y_train):
                label_indices = np.where(y_train == label)[0]
                label_quota = max(1, int(sample_size * len(label_indices) / len(y_train)))
                sampled_indices.extend(
                    rng.choice(
                        label_indices,
                        size=min(label_quota, len(label_indices)),
                        replace=False,
                    ).tolist()
                )
            if len(sampled_indices) < sample_size:
                remaining = np.setdiff1d(np.arange(len(X_train)), np.array(sampled_indices))
                sampled_indices.extend(
                    rng.choice(
                        remaining,
                        size=min(sample_size - len(sampled_indices), len(remaining)),
                        replace=False,
                    ).tolist()
                )
            sampled_indices = np.array(sampled_indices[:sample_size])
            X_train = X_train[sampled_indices]
            y_train = y_train[sampled_indices]
            logger.info(
                "    Using representative fitness sample: "
                f"{len(X_train)} rows, class distribution={np.bincount(y_train).tolist()}"
            )
        return X_train, y_train

    def _run_evolution(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Run evolutionary algorithm"""
        logger.info("\n[3] Running evolutionary algorithm...")

        # Create fitness function
        def evaluate_individual(individual, X, y):
            try:
                func = gp.compile(expr=individual, pset=self.pset)
            except Exception:
                # If compilation fails, return worst fitness
                return (0.0,)
            
            predictions = []
            for sample in X:
                try:
                    pred = func(*sample)
                    pred = protected_logistic(pred)
                    predictions.append(pred)
                except Exception:
                    predictions.append(0.5)
            
            predictions = np.array(predictions)
            binary_preds = (predictions > 0.5).astype(int)
            
            # F1 score as fitness
            try:
                f1 = f1_score(y, binary_preds, average="weighted", zero_division=0)
            except:
                f1 = 0.0
            
            return (f1,)

        self.toolbox.register("evaluate", evaluate_individual, X=X_train, y=y_train)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=GENETIC_PROGRAMMING_CONFIG["tournament_size"])

        # Size limits
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GENETIC_PROGRAMMING_CONFIG["max_depth"]))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GENETIC_PROGRAMMING_CONFIG["max_depth"]))

        # Create initial population
        pop = self.toolbox.population(n=GENETIC_PROGRAMMING_CONFIG["population_size"])

        # Evaluate initial population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Run algorithm
        pop, logbook = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=GENETIC_PROGRAMMING_CONFIG["cx_probability"],
            mutpb=GENETIC_PROGRAMMING_CONFIG["mut_probability"],
            ngen=GENETIC_PROGRAMMING_CONFIG["generations"],
            stats=None,
            verbose=False,
        )

        self.best_individual = tools.selBest(pop, 1)[0]
        self.logbook = logbook

        best_fitness = self.best_individual.fitness.values[0]
        logger.info(f"    Best F1 score: {best_fitness:.4f}")
        logger.info(f"    Best tree depth: {self.best_individual.height}")
        logger.info(f"    Best tree size: {len(self.best_individual)}")

    def _evaluate_best(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Evaluate best individual"""
        try:
            func = gp.compile(expr=self.best_individual, pset=self.pset)
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            return 0.0
        
        predictions = []
        
        for sample in X_train:
            try:
                predictions.append(protected_logistic(func(*sample)))
            except Exception:
                predictions.append(0.5)
        
        predictions = np.array(predictions)
        binary_preds = (predictions > 0.5).astype(int)
        
        f1 = f1_score(y_train, binary_preds, average="weighted", zero_division=0)
        precision = precision_score(y_train, binary_preds, average="weighted", zero_division=0)
        recall = recall_score(y_train, binary_preds, average="weighted", zero_division=0)
        
        logger.info(f"\n[EVALUATION]")
        logger.info(f"    F1: {f1:.4f}")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall: {recall:.4f}")
        
        return f1

    def _save_results(self) -> None:
        """Save GP results"""
        logger.info("\n[4] Saving GP results...")

        # Save best tree
        model_path = OUTPUTS_DIR / "best_tree.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.best_individual, f)
        logger.info(f"    Saved best tree to: {model_path}")

        bundle_path = OUTPUTS_DIR / "saligp_igp_model.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(
                {
                    "tree": str(self.best_individual),
                    "features": ALL_FEATURES,
                    "fitness": float(self.best_fitness),
                    "threshold": 0.5,
                    "model_type": "DEAP Improved Genetic Programming Tree",
                },
                f,
            )
        logger.info(f"    Saved loadable IGP model bundle to: {bundle_path}")

        metadata_path = OUTPUTS_DIR / "saligp_igp_model.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "tree": str(self.best_individual),
                    "features": ALL_FEATURES,
                    "fitness": float(self.best_fitness),
                    "threshold": 0.5,
                    "model_type": "DEAP Improved Genetic Programming Tree",
                },
                f,
                indent=2,
            )
        logger.info(f"    Saved IGP metadata to: {metadata_path}")

        # Save tree as text
        rule_path = OUTPUTS_DIR / "best_rule.txt"
        with open(rule_path, "w") as f:
            f.write(str(self.best_individual))
        logger.info(f"    Saved rule to: {rule_path}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ GENETIC PROGRAMMING COMPLETE")
        logger.info("=" * 60)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using evolved GP rule"""
        if self.pset is None:
            self._setup_deap()
        if self.best_individual is None:
            logger.error("Prediction requested before IGP model has been trained or loaded.")
            return np.full(len(X), 0.5)
        try:
            func = gp.compile(expr=self.best_individual, pset=self.pset)
        except Exception as e:
            logger.error(f"Prediction compilation failed: {e}")
            return np.full(len(X), 0.5)
        
        predictions = []
        
        for sample in X:
            try:
                predictions.append(protected_logistic(func(*sample)))
            except Exception:
                predictions.append(0.5)
        
        return np.array(predictions)

    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        proba = self.predict(X)
        return (proba > threshold).astype(int)

    def get_best_tree(self):
        """Get best evolved tree"""
        return self.best_individual

    def load(self, model_path: Optional[Path] = None) -> "ImprovedGeneticProgramming":
        """Load a persisted IGP model bundle."""
        if self.pset is None:
            self._setup_deap()

        path = model_path or (OUTPUTS_DIR / "saligp_igp_model.pkl")
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        tree_text = bundle["tree"] if isinstance(bundle, dict) else str(bundle)
        self.best_individual = gp.PrimitiveTree.from_string(tree_text, self.pset)
        self.best_fitness = float(bundle.get("fitness", 0.0)) if isinstance(bundle, dict) else 0.0
        return self

