import numpy as np

DEFAULT_THRESHOLD: float = 0.40
RANDOM_STATE: int = 42
SEMAGESTAC_PRETERM_CUTOFF: int = 37

GRAVIDEZ_COLUMN: str = "GRAVIDEZ"
GRAVIDEZ_SINGLE_VALUE: int = 1

MESPRENAT_COLUMN: str = "MESPRENAT"
ESCMAE2010_COLUMN: str = "ESCMAE2010"
KOTELCHUCK_COLUMN: str = "KOTELCHUCK"
SEMAGESTAC_COLUMN: str = "SEMAGESTAC"
IDADEMAE_COLUMN: str = "IDADEMAE"
QTDFILMORT_COLUMN: str = "QTDFILMORT"
QTDPARTNOR_COLUMN: str = "QTDPARTNOR"
QTDPARTCES_COLUMN: str = "QTDPARTCES"

RAW_COLUMNS: list[str] = [
    SEMAGESTAC_COLUMN,
    "GESTACAO",
    IDADEMAE_COLUMN,
    ESCMAE2010_COLUMN,
    "ESTCIVMAE",
    "RACACORMAE",
    "QTDGESTANT",
    QTDPARTNOR_COLUMN,
    QTDPARTCES_COLUMN,
    "QTDFILVIVO",
    QTDFILMORT_COLUMN,
    GRAVIDEZ_COLUMN,
    MESPRENAT_COLUMN,
    "CONSPRENAT",
    "SEXO",
    KOTELCHUCK_COLUMN,
    "IDADEPAI",
    "LATITUDE",
    "LONGITUDE",
    "PREMATURO",
    "PAI_AUSENTE",
    "IDADEPAI_INVALIDA",
]

LEAKAGE_COLUMNS: list[str] = [
    "SEMAGESTAC",
    "GESTACAO",
    "ESTCIVMAE",
    "RACACORMAE",
    "SEXO",
    "CONSPRENAT",
    "IDADEPAI",
    "IDADEPAI_INVALIDA",
]

NUMERIC_COLUMNS: list[str] = [
    "IDADEMAE",
    "QTDGESTANT",
    "QTDPARTNOR",
    "QTDPARTCES",
    "QTDFILVIVO",
    "QTDFILMORT",
    "MESPRENAT",
    "LATITUDE",
    "LONGITUDE",
]

PASSTHROUGH_COLUMNS: list[str] = [
    "PAI_AUSENTE",
]

SENTINEL_REPLACEMENTS: dict[str, tuple[object, object]] = {
    MESPRENAT_COLUMN: (99, np.nan),
    ESCMAE2010_COLUMN: (9, np.nan),
    KOTELCHUCK_COLUMN: ("9", np.nan),
    GRAVIDEZ_COLUMN: (9, np.nan),
}

IDADEMAE_MIN: int = 10
IDADEMAE_MAX: int = 60
MESPRENAT_MIN: int = 1
MESPRENAT_MAX: int = 12

"""
SEMAGESTAC_MIN: int = 0
SEMAGESTAC_MAX: int = 45
"""

VALID_ESCMAE2010_VALUES: set[int | float] = {0, 1, 2, 3, 4, 5}
VALID_KOTELCHUCK_VALUES: set[str] = {"1", "2", "3", "4", "5"}

KOTELCHUCK_MAP: dict[str, int | None] = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "9": None,
}

ESCMAE2010_MAP: dict[float, int | None] = {
    0.0: 0,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    4.0: 4,
    5.0: 5,
    9.0: None,
}

ESCMAE2010_ORDINAL_COLUMN: str = "ESCMAE2010_ORDINAL"
KOTELCHUCK_ORDINAL_COLUMN: str = "KOTELCHUCK_ORDINAL"

COUNT_COLUMNS: list[str] = [
    "QTDGESTANT",
    QTDPARTNOR_COLUMN,
    QTDPARTCES_COLUMN,
    "QTDFILVIVO",
    QTDFILMORT_COLUMN,
]

HISTGB_PARAMS: dict[str, float | int | None] = {
    "learning_rate": 0.2,
    "max_iter": 100,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_depth": None,
    "max_bins": 255,
}
