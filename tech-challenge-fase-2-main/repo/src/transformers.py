from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.constants import (
    ESCMAE2010_COLUMN,
    ESCMAE2010_MAP,
    KOTELCHUCK_COLUMN,
    KOTELCHUCK_MAP,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering transformer.

    Produces derived ordinal features:
    - ESCMAE2010_ORDINAL: mapped from ESCMAE2010 via ESCMAE2010_MAP
    - KOTELCHUCK_ORDINAL: mapped from KOTELCHUCK via KOTELCHUCK_MAP
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame as input")

        out = X.copy()

        if ESCMAE2010_COLUMN in out.columns:
            out["ESCMAE2010_ORDINAL"] = out[ESCMAE2010_COLUMN].map(ESCMAE2010_MAP)

        if KOTELCHUCK_COLUMN in out.columns:
            out["KOTELCHUCK_ORDINAL"] = out[KOTELCHUCK_COLUMN].map(KOTELCHUCK_MAP)

        return out

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> list[str]:
        base = list(input_features) if input_features is not None else []
        derived = ["ESCMAE2010_ORDINAL", "KOTELCHUCK_ORDINAL"]
        return base + [c for c in derived if c not in base]
