"""Simple scenario-based sklearn pipeline for the modeling workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_DATA_DIR = Path("data")
DEFAULT_METRICS_DIR = Path("results/metrics")
REQUIRED_DATA_FILES = ("X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet")
REQUIRED_METRIC_FILES = (
    "experiment_config.json",
    "model_comparison_cv_test.csv",
    "best_model_operational_metrics.json",
)
SCENARIO_CHOICES = ("A", "B", "C")
MODEL_CHOICES = ("logreg", "random_forest", "histgb")
NOTEBOOK_MODEL_NAMES = ("LogReg", "RandomForest", "HistGB")
MODEL_DISPLAY_NAMES = {
    "logreg": "LogisticRegression",
    "random_forest": "RandomForest",
    "histgb": "HistGradientBoosting + sample_weight",
}
SCENARIO_DESCRIPTIONS = {
    "A": "dummies + bools",
    "B": "ordinal + continuo",
    "C": "todos",
}
BASE_NUMERIC_COLUMNS = [
    "IDADEMAE",
    "QTDGESTANT",
    "QTDPARTNOR",
    "QTDPARTCES",
    "QTDFILVIVO",
    "QTDFILMORT",
    "MESPRENAT",
    "LATITUDE",
    "LONGITUDE",
    "PAI_AUSENTE",
    "KOTELCHUCK_IGNORADO",
    "ESCMAE2010_IGNORADO",
    "PNTARDIO",
    "HISTPERDAFETAL",
    "PRIMIPARA",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run simple sklearn pipelines for modeling scenarios."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("validate", help="Validate required artifacts.")
    subparsers.add_parser("summarize", help="Summarize current baseline artifacts.")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one scenario/model combination."
    )
    evaluate_parser.add_argument("--scenario", choices=SCENARIO_CHOICES, required=True)
    evaluate_parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    evaluate_parser.add_argument("--output-json", type=Path, default=None)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple scenario/model combinations."
    )
    compare_parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_CHOICES,
        default=list(SCENARIO_CHOICES),
    )
    compare_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["logreg", "random_forest", "histgb"],
    )
    compare_parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/metrics/pipeline_comparison.csv"),
    )
    return parser


def validate_inputs(data_dir: Path, metrics_dir: Path) -> None:
    missing_paths: list[str] = []

    for filename in REQUIRED_DATA_FILES:
        path = data_dir / filename
        if not path.exists():
            missing_paths.append(str(path))

    if not metrics_dir.exists():
        missing_paths.append(str(metrics_dir))
    else:
        for filename in REQUIRED_METRIC_FILES:
            path = metrics_dir / filename
            if not path.exists():
                missing_paths.append(str(path))

    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Required pipeline artifacts are missing.\n"
            "Run the notebook flow first and try again.\n"
            f"{formatted_paths}"
        )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def summarize_current_baseline(metrics_dir: Path) -> dict[str, Any]:
    experiment_config = load_json(metrics_dir / "experiment_config.json")
    comparison_rows = load_comparison_rows(metrics_dir / "model_comparison_cv_test.csv")
    operational_metrics = load_json(metrics_dir / "best_model_operational_metrics.json")
    best_cv_row = max(
        comparison_rows,
        key=lambda row: (float(row["cv_f2_mean"]), float(row["cv_recall_mean"])),
    )
    tuned_metrics = operational_metrics.get("metrics", {})

    return {
        "random_state": experiment_config.get("random_state"),
        "cv_splits": experiment_config.get("cv_splits"),
        "min_recall_threshold": experiment_config.get("min_recall_threshold"),
        "best_cv_model": best_cv_row.get("model"),
        "best_cv_f2_mean": float(best_cv_row["cv_f2_mean"]),
        "best_cv_recall_mean": float(best_cv_row["cv_recall_mean"]),
        "operational_model": operational_metrics.get("model"),
        "operational_threshold": operational_metrics.get("threshold"),
        "operational_recall": tuned_metrics.get("recall"),
        "operational_f2": tuned_metrics.get("f2"),
        "selection_rule": operational_metrics.get("selection_rule"),
    }


def print_mapping(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key, value in values.items():
        print(f"- {key}: {value}")


def build_scenario_model_grid(
    scenarios: list[str] | tuple[str, ...] | None = None,
    models: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    selected_scenarios = scenarios or list(SCENARIO_CHOICES)
    selected_models = models or list(MODEL_CHOICES)
    return [
        {"scenario": scenario, "model": model_name}
        for scenario in selected_scenarios
        for model_name in selected_models
    ]


def build_scenario_catalog(
    scenarios: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    selected_scenarios = scenarios or list(SCENARIO_CHOICES)
    return [
        {
            "scenario": scenario,
            "description": SCENARIO_DESCRIPTIONS[scenario],
        }
        for scenario in selected_scenarios
    ]


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = pd.read_parquet(data_dir / "X_train.parquet")
    x_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(data_dir / "y_test.parquet").iloc[:, 0]
    return x_train, x_test, y_train, y_test



_SCENARIO_A_DROP = [
    "IDADEMAE", "MESPRENAT", "QTDFILMORT", "QTDFILVIVO", "QTDGESTANT",
    "ESCMAE2010_ORDINAL", "KOTELCHUCK_ORDINAL",
]
_SCENARIO_B_DROP = ["PNTARDIO", "HISTPERDAFETAL", "PRIMIPARA"]


def apply_scenario(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "C":
        return frame

    updated = frame.copy()

    if scenario == "A":
        drop = [c for c in _SCENARIO_A_DROP if c in updated.columns]
        return updated.drop(columns=drop)

    if scenario == "B":
        faixa_cols = [c for c in updated.columns if c.startswith("FAIXAETAMAE_")]
        kotk_dummies = [c for c in updated.columns if c.startswith("KOTELCHUCK_") and c != "KOTELCHUCK_ORDINAL" and c != "KOTELCHUCK_IGNORADO"]
        esc_dummies = [c for c in updated.columns if c.startswith("ESCMAE2010_") and c != "ESCMAE2010_ORDINAL" and c != "ESCMAE2010_IGNORADO"]
        drop = faixa_cols + kotk_dummies + esc_dummies + [c for c in _SCENARIO_B_DROP if c in updated.columns]
        return updated.drop(columns=drop)

    raise ValueError(f"Unsupported scenario: {scenario}")


def get_numeric_columns(frame: pd.DataFrame) -> list[str]:
    extra = [c for c in ("ESCMAE2010_ORDINAL", "KOTELCHUCK_ORDINAL") if c in frame.columns]
    return [c for c in BASE_NUMERIC_COLUMNS + extra if c in frame.columns]


def build_estimator(model_name: str) -> Any:
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    if model_name == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_fit_kwargs(
    y: pd.Series,
    use_balanced_sample_weight: bool = False,
    pipeline_step_name: str | None = None,
) -> dict[str, Any]:
    if not use_balanced_sample_weight:
        return {}

    weights = compute_sample_weight(class_weight="balanced", y=y)
    if pipeline_step_name is None:
        return {"sample_weight": weights}
    return {f"{pipeline_step_name}__sample_weight": weights}


def build_model_search_configs(random_state: int = 42, n_jobs: int = -1) -> dict[str, dict[str, Any]]:
    return {
        "LogReg": {
            "pipeline_model_name": "logreg",
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=random_state,
            ),
            "param_distributions": [
                {"C": np.logspace(-2, 1, 8), "solver": ["liblinear"], "penalty": ["l1", "l2"]},
                {"C": np.logspace(-2, 1, 8), "solver": ["lbfgs"],     "penalty": ["l2"]},
            ],
            "n_iter": 10,
            "imbalance_strategy": "class_weight=balanced",
            "use_balanced_sample_weight": False,
        },
        "RandomForest": {
            "pipeline_model_name": "random_forest",
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            "param_distributions": {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, 16],
                "min_samples_leaf": [2, 4, 8],
                "max_features": ["sqrt", "log2", 0.5],
                "class_weight": ["balanced_subsample"],
            },
            "n_iter": 10,
            "imbalance_strategy": "random_search(class_weight)",
            "use_balanced_sample_weight": False,
        },
        "HistGB": {
            "pipeline_model_name": "histgb",
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "param_distributions": {
                "max_iter": [150, 250, 350],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
            "n_iter": 10,
            "imbalance_strategy": "sample_weight=balanced",
            "use_balanced_sample_weight": True,
        },
    }


def build_randomized_search(
    estimator: Any,
    param_distributions: dict[str, Any],
    scoring: dict[str, Any],
    cv: Any,
    n_iter: int,
    random_state: int = 42,
    n_jobs: int = -1,
    refit: str = "f2",
) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )


def build_pipeline(frame: pd.DataFrame, model_name: str):
    if model_name == "logreg":
        numeric_columns = get_numeric_columns(frame)
        preprocessor = make_column_transformer(
            (StandardScaler(), numeric_columns),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return make_pipeline(preprocessor, build_estimator(model_name))

    return make_pipeline(build_estimator(model_name))


def fit_pipeline(pipeline, x_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> None:
    fit_params = build_fit_kwargs(
        y_train,
        use_balanced_sample_weight=(model_name == "histgb"),
        pipeline_step_name="histgradientboostingclassifier" if model_name == "histgb" else None,
    )
    pipeline.fit(x_train, y_train, **fit_params)


def get_scores(pipeline, x_test: pd.DataFrame) -> pd.Series:
    if hasattr(pipeline, "predict_proba"):
        return pd.Series(pipeline.predict_proba(x_test)[:, 1], index=x_test.index)
    if hasattr(pipeline, "decision_function"):
        return pd.Series(pipeline.decision_function(x_test), index=x_test.index)
    return pd.Series(pipeline.predict(x_test), index=x_test.index)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> dict[str, float]:
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def run_evaluation(data_dir: Path, scenario: str, model_name: str) -> dict[str, Any]:
    x_train, x_test, y_train, y_test = load_datasets(data_dir)
    x_train = apply_scenario(x_train, scenario)
    x_test = apply_scenario(x_test, scenario)

    pipeline = build_pipeline(x_train, model_name)
    fit_pipeline(pipeline, x_train, y_train, model_name)

    y_pred = pd.Series(pipeline.predict(x_test), index=y_test.index)
    y_score = get_scores(pipeline, x_test)
    metrics = evaluate_predictions(y_test, y_pred, y_score)

    return {
        "scenario": scenario,
        "model": model_name,
        "n_features": x_train.shape[1],
        **metrics,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)
        file.write("\n")


def write_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    command = args.command or "summarize"

    validate_inputs(args.data_dir, args.metrics_dir)

    if command == "validate":
        print("Pipeline validation: OK")
        print(f"- data_dir: {args.data_dir}")
        print(f"- metrics_dir: {args.metrics_dir}")
        return

    if command == "summarize":
        print_mapping("Current baseline summary:", summarize_current_baseline(args.metrics_dir))
        return

    if command == "evaluate":
        result = run_evaluation(args.data_dir, args.scenario, args.model)
        print_mapping("Evaluation result:", result)
        if args.output_json is not None:
            write_json(result, args.output_json)
        return

    if command == "compare":
        rows: list[dict[str, Any]] = []
        for scenario in args.scenarios:
            for model_name in args.models:
                rows.append(run_evaluation(args.data_dir, scenario, model_name))
        write_comparison_csv(rows, args.output_csv)
        print(f"Saved comparison CSV: {args.output_csv}")
        return

    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    main()
