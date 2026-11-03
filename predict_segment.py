#!/usr/bin/env python3
"""Customer segment prediction - inference only.

Loads the model saved by customer_segmentation.py (joblib) and assigns new
customers to a segment. No training happens here.

Usage:
    python predict_segment.py                       (scores new_customers.csv)
    python predict_segment.py new_customers.csv
    python predict_segment.py new_customers.csv --output scored_customers.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import pandas as pd
import sklearn

LOGGER = logging.getLogger("segment_predict")

DEFAULT_MODEL_PATH = Path("outputs/customer_segmentation_model.joblib")
DEFAULT_METADATA_PATH = Path("outputs/model_metadata.json")
DEFAULT_INPUT_PATH = Path("new_customers.csv")


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read the model card written during training."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run customer_segmentation.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    """Load the saved pipeline (scaler + K-Means) from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run customer_segmentation.py first."
        )
    return joblib.load(path)


def warn_if_version_differs(metadata: Dict[str, Any]) -> None:
    """A model should be loaded with the scikit-learn version that built it."""
    trained_with = metadata.get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        LOGGER.warning(
            "Model was trained with scikit-learn %s but %s is installed. "
            "Results may be wrong or loading may fail.",
            trained_with,
            sklearn.__version__,
        )


def read_customers(path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Read customers from CSV and check they match what the model expects."""
    if not path.exists():
        raise FileNotFoundError(f"Customer file not found: {path}")

    data = pd.read_csv(path)
    features = metadata["features"]

    missing_columns = [c for c in features if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    for column in features:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must contain numbers only")

    empty_rows = data.index[data[features].isnull().any(axis=1)].tolist()
    if empty_rows:
        raise ValueError(f"Rows with missing values (0-based row numbers): {empty_rows}")

    # A K-Means model still assigns values it never saw, but the result is less reliable
    for column, (low, high) in metadata["training_ranges"].items():
        outside = data.index[(data[column] < low) | (data[column] > high)].tolist()
        if outside:
            LOGGER.warning(
                "%s outside the training range %s-%s in rows %s; treat these results with care.",
                column, low, high, outside,
            )
    return data


def score_customers(model, customers: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Assign each customer to a segment and flag unusual customers."""
    features = metadata["features"]
    clusters = model.predict(customers[features])
    distances = model.transform(customers[features]).min(axis=1)

    segments = metadata["segments"]
    result = customers.copy()
    result["Segment"] = [segments[str(c)]["segment"] for c in clusters]
    result["Persona"] = [segments[str(c)]["persona"] for c in clusters]
    result["DistanceToCentroid"] = distances.round(2)
    result["Unusual"] = distances > metadata["unusual_distance_limit"]
    result["RecommendedAction"] = [segments[str(c)]["action"] for c in clusters]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_segment.py",
        description="Assign new customers to a segment with the saved model (no retraining).",
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT_PATH,
                        help=f"CSV file with the customer columns (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="joblib model file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH,
                        help="model metadata JSON")
    parser.add_argument("--output", type=Path, default=None, help="save scored customers to CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    try:
        metadata = load_metadata(args.metadata)
        warn_if_version_differs(metadata)
        model = load_model(args.model)
        customers = read_customers(args.input, metadata)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1

    result = score_customers(model, customers, metadata)
    LOGGER.info("Assigned %d customer(s) to segments", len(result))
    print(result[["Segment", "Persona", "DistanceToCentroid", "Unusual"]].to_string())

    print("\nRecommended actions:")
    for persona, action in result.drop_duplicates("Persona")[["Persona", "RecommendedAction"]].values:
        print(f"  {persona}: {action}")

    if args.output:
        result.to_csv(args.output, index=False)
        LOGGER.info("Saved scored customers to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
