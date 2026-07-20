
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import (
    ESCMAE2010_ORDINAL_COLUMN,
    HISTGB_PARAMS,
    KOTELCHUCK_ORDINAL_COLUMN,
    NUMERIC_COLUMNS,
    PASSTHROUGH_COLUMNS,
    RANDOM_STATE,
)
from src.transformers import FeatureEngineer


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    numeric_columns = list(NUMERIC_COLUMNS) + [
        ESCMAE2010_ORDINAL_COLUMN,
        KOTELCHUCK_ORDINAL_COLUMN,
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("passthrough", "passthrough", list(PASSTHROUGH_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(random_state=random_state, **HISTGB_PARAMS)

    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("histgradientboostingclassifier", classifier),
        ]
    )
