"""
Production ASGI entrypoint for Render.

This loads the persisted DEAP IGP tree model and exposes the FastAPI app.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SALIGP_ROOT = PROJECT_ROOT / "saligp"
if str(SALIGP_ROOT) not in sys.path:
    sys.path.insert(0, str(SALIGP_ROOT))

from api.server import SALIGPAPIServer
from genetic_programming import ImprovedGeneticProgramming
from pipeline import SALIGPClassifier


def create_app():
    model_path = Path(
        os.getenv("SALIGP_MODEL_PATH", str(SALIGP_ROOT / "outputs" / "saligp_igp_model.pkl"))
    )
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(
            "SALIGP IGP model artifact is missing. "
            f"Expected a non-empty model at: {model_path}"
        )

    gp_model = ImprovedGeneticProgramming(data_loader=None).load(model_path)
    classifier = SALIGPClassifier(
        gp_model=gp_model,
        uncertainty_scores={},
        cluster_assignments=np.array([]),
    )
    return SALIGPAPIServer(classifier).get_app()


app = create_app()
