"""Run the shared training pipeline on a local CSV file."""

import argparse
import json
from pathlib import Path

from diabetes_model.contract import DATASET_SCHEMA_VERSION
from diabetes_model.features import FEATURE_COLUMNS, TRANSFORMATION_VERSION
from diabetes_model.models import CANDIDATES, PRIMARY_METRIC
from diabetes_model.training import best_candidate, prepare_dataset, train_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to Diabetes.csv")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATES,
        default=list(CANDIDATES),
        help="candidate models to train",
    )
    args = parser.parse_args()

    data = prepare_dataset(args.dataset)
    results = train_candidates(data, tuple(args.candidates))
    result = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "transformation_version": TRANSFORMATION_VERSION,
        "eligible_rows": len(data.target),
        "rejected_rows": data.rejected_rows,
        "excluded_rows": data.excluded_rows,
        "feature_count": len(FEATURE_COLUMNS),
        "split_rows": data.split.value_counts().to_dict(),
        "primary_selection_metric": PRIMARY_METRIC,
        "best_candidate": best_candidate(results),
        "candidates": {name: metrics for name, (_, metrics) in results.items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
