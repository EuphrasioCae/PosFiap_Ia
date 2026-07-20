from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    ESCMAE2010_COLUMN,
    GRAVIDEZ_COLUMN,
    GRAVIDEZ_SINGLE_VALUE,
    KOTELCHUCK_COLUMN,
    LEAKAGE_COLUMNS,
    MESPRENAT_COLUMN,
    SENTINEL_REPLACEMENTS,
    VALID_ESCMAE2010_VALUES,
    VALID_KOTELCHUCK_VALUES,
    IDADEMAE_COLUMN,
    IDADEMAE_MAX,
    IDADEMAE_MIN,
    MESPRENAT_MAX,
    MESPRENAT_MIN,
    SEMAGESTAC_COLUMN,
    QTDFILMORT_COLUMN,
    QTDPARTCES_COLUMN,
    QTDPARTNOR_COLUMN,
)


def _replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    for column, (sentinel, replacement) in SENTINEL_REPLACEMENTS.items():
        if column not in df.columns:
            continue
        df.loc[df[column] == sentinel, column] = replacement
    return df


def _validate_plausibility(df: pd.DataFrame) -> pd.DataFrame:
    if IDADEMAE_COLUMN in df.columns:
        df.loc[
            ~df[IDADEMAE_COLUMN].between(IDADEMAE_MIN, IDADEMAE_MAX, inclusive="both"),
            IDADEMAE_COLUMN,
        ] = np.nan

    if MESPRENAT_COLUMN in df.columns:
        df.loc[
            ~df[MESPRENAT_COLUMN].between(MESPRENAT_MIN, MESPRENAT_MAX, inclusive="both"),
            MESPRENAT_COLUMN,
        ] = np.nan

    if ESCMAE2010_COLUMN in df.columns:
        df.loc[
            ~df[ESCMAE2010_COLUMN].isin(VALID_ESCMAE2010_VALUES),
            ESCMAE2010_COLUMN,
        ] = np.nan

    if KOTELCHUCK_COLUMN in df.columns:
        df.loc[
            ~df[KOTELCHUCK_COLUMN].isin(VALID_KOTELCHUCK_VALUES),
            KOTELCHUCK_COLUMN,
        ] = np.nan

    from src.constants import COUNT_COLUMNS

    for col in COUNT_COLUMNS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    
    # if SEMAGESTAC_COLUMN in df.columns:
    #     df.loc[
    #         ~df[SEMAGESTAC_COLUMN].between(SEMAGESTAC_MIN, SEMAGESTAC_MAX, inclusive="both"),
    #         SEMAGESTAC_COLUMN,
    #     ] = np.nan
    
    
    return df


def _filter_gravidez(df: pd.DataFrame) -> pd.DataFrame:
    if GRAVIDEZ_COLUMN not in df.columns:
        return df
    filtered = df[df[GRAVIDEZ_COLUMN] == GRAVIDEZ_SINGLE_VALUE].copy()
    return filtered


def _drop_unsupported_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        col for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]

    return df.drop(columns=columns_to_drop, errors="ignore")


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw SINASC dataset for offline training.

    The function applies the offline preparation rules defined for the
    Fase 2 pipeline:
      - keep only single pregnancies (GRAVIDEZ == 1)
      - convert sentinel values to NaN
      - validate plausibility of main fields
      - remove duplicate records
      - drop columns outside the modeling scope

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw input dataset loaded from `data/df_model_raw.parquet`.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the dataset with `GRAVIDEZ` removed.
    """
    df_clean = df_raw.copy()
    df_clean = _replace_sentinels(df_clean)
    df_clean = _filter_gravidez(df_clean)
    df_clean = _validate_plausibility(df_clean)
    df_clean = df_clean.drop(columns=[GRAVIDEZ_COLUMN], errors="ignore")
    df_clean = df_clean.drop_duplicates()
    df_clean = _drop_unsupported_columns(df_clean)
    return df_clean
