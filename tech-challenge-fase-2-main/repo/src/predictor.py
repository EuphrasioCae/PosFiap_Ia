from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DEFAULT_THRESHOLD
from src.schemas import FatorSHAP, RequisicaoPredicao, RespostaPredicao

_FEATURE_LABELS: dict[str, str] = {
    "IDADEMAE":             "Idade da mãe",
    "QTDGESTANT":           "Gestações anteriores",
    "QTDPARTNOR":           "Partos normais",
    "QTDPARTCES":           "Partos cesáreos",
    "QTDFILVIVO":           "Filhos vivos",
    "QTDFILMORT":           "Filhos mortos",
    "MESPRENAT":            "Mês de início do pré-natal",
    "LATITUDE":             "Latitude do local de nascimento",
    "LONGITUDE":            "Longitude do local de nascimento",
    "ESCMAE2010_ORDINAL":   "Escolaridade da mãe",
    "KOTELCHUCK_ORDINAL":   "Adequação do pré-natal (Kotelchuck)",
    "PAI_AUSENTE":          "Pai ausente na declaração de nascimento",
}

# Mapeamento de nome pós-preprocessamento → campo original da requisição
_FEATURE_TO_RAW: dict[str, str] = {
    "ESCMAE2010_ORDINAL": "ESCMAE2010",
    "KOTELCHUCK_ORDINAL": "KOTELCHUCK",
}

# Binary flags whose local effect must follow the calibrated probability, not TreeSHAP.
# TreeSHAP explains the uncalibrated HistGB; isotonic calibration can invert small effects.
_COUNTERFACTUAL_BINARY_FEATURES: dict[str, str] = {
    "PAI_AUSENTE": "PAI_AUSENTE",
}

# Limites empíricos derivados da distribuição de probabilidade no conjunto de teste
_MODEL_PROB_CEILING: float = 0.89   # p99 dos prematuros reais
_MODEL_PROB_CONCERN: float = 0.35   # p5  dos prematuros reais — abaixo disso, baixo risco genuíno

_WARNINGS_STATIC = [
    "Modelo treinado com dados do SINASC (MG, 2020–2022). Não substitui avaliação clínica.",
    "Recall operacional: 83% — cerca de 17% dos prematuros reais ficam abaixo do threshold (falsos negativos).",
    f"Escala de referência do modelo: threshold={DEFAULT_THRESHOLD} | teto prático={_MODEL_PROB_CEILING}. "
    "Probabilidades acima de 0.70 representam o decil superior de risco.",
    "Pai ausente usa efeito contrafactual na probabilidade calibrada (valor atual vs oposto); "
    "demais fatores usam SHAP do modelo base antes da calibração isotônica.",
]


def _prob_context_warning(prob: float) -> str:
    """Gera nota contextual para a LLM interpretar a probabilidade corretamente."""
    if prob >= DEFAULT_THRESHOLD:
        span = _MODEL_PROB_CEILING - DEFAULT_THRESHOLD
        pct  = min(100, round((prob - DEFAULT_THRESHOLD) / span * 100))
        return (
            f"Probabilidade {prob:.3f} → {pct}% da margem disponível entre threshold "
            f"({DEFAULT_THRESHOLD}) e teto (~{_MODEL_PROB_CEILING}). "
            f"Distância ao teto: {_MODEL_PROB_CEILING - prob:.3f}."
        )
    if prob >= _MODEL_PROB_CONCERN:
        return (
            f"Probabilidade {prob:.3f} está abaixo do threshold ({DEFAULT_THRESHOLD}), mas dentro da zona cinza "
            f"({_MODEL_PROB_CONCERN}–{DEFAULT_THRESHOLD}): ~12% dos prematuros reais têm probabilidade nessa faixa. "
            "Baixo risco operacional, mas não risco zero — considerar contexto clínico."
        )
    return (
        f"Probabilidade {prob:.3f} está abaixo de {_MODEL_PROB_CONCERN} (p5 dos prematuros reais). "
        "Risco genuinamente baixo para este modelo."
    )


class PredictorPrematuro:
    """Carrega modelo calibrado + SHAP bundle e expõe predict().

    Uso em Streamlit:
        @st.cache_resource
        def load_predictor():
            return PredictorPrematuro.from_artifacts_dir("results/artifacts")

        predictor = load_predictor()
        resposta  = predictor.predict(req)
    """

    def __init__(self, model_path: str | Path, shap_bundle_path: str | Path) -> None:
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(shap_bundle_path, "rb") as f:
            bundle = pickle.load(f)
        self._explainer     = bundle["explainer"]
        self._preprocessor  = bundle["preprocessor"]
        self._feature_names = bundle["feature_names"]

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir: str | Path) -> "PredictorPrematuro":
        d = Path(artifacts_dir)
        return cls(
            model_path=d / "model_phase2.pkl",
            shap_bundle_path=d / "shap_bundle.pkl",
        )

    def predict(self, req: RequisicaoPredicao) -> RespostaPredicao:
        raw  = req.model_dump()
        X    = pd.DataFrame([raw])

        prob = float(self._model.predict_proba(X)[0, 1])

        shap_arr = self._shap_values(X)
        counterfactual = self._counterfactual_binary_contributions(raw)

        fatores = []
        for fname, sv in zip(self._feature_names, shap_arr):
            raw_key = _FEATURE_TO_RAW.get(fname, fname)
            raw_value = raw.get(raw_key)
            if fname in counterfactual:
                sv = counterfactual[fname]
                note = _clinical_note_counterfactual(fname, raw_value, sv)
            else:
                note = _clinical_note(fname, raw_value, sv)
            fatores.append(
                FatorSHAP(
                    feature=fname,
                    label=_FEATURE_LABELS.get(fname, fname),
                    raw_value=raw_value,
                    shap_value=round(float(sv), 6),
                    clinical_note=note,
                )
            )

        top_risk = _top(fatores, positive=True)
        top_prot = _top(fatores, positive=False)

        return RespostaPredicao(
            risk_probability=round(prob, 4),
            risk_label="alto_risco_operacional" if prob >= DEFAULT_THRESHOLD else "baixo_risco",
            clinical_risk_level=_clinical_risk_level(prob),
            threshold=DEFAULT_THRESHOLD,
            margin_to_threshold=round(prob - DEFAULT_THRESHOLD, 4),
            top_risk_factors=top_risk or None,
            top_protective_factors=top_prot or None,
            interpretation_warnings=_WARNINGS_STATIC + [_prob_context_warning(prob)],
        )

    def _shap_values(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self._preprocessor.transform(X)
        X_df   = pd.DataFrame(X_prep, columns=self._feature_names)
        raw    = self._explainer.shap_values(X_df, check_additivity=False)

        # TreeExplainer pode retornar: lista [cls0, cls1], array 3D (n, f, 2) ou 2D (n, f)
        if isinstance(raw, list):
            arr = np.asarray(raw[1])          # classe positiva
        else:
            arr = np.asarray(raw)
            if arr.ndim == 3:
                arr = arr[:, :, 1]

        return arr[0]                          # única amostra → (n_features,)

    def _counterfactual_binary_contributions(self, raw: dict) -> dict[str, float]:
        """Marginal effect on calibrated probability vs the opposite binary value."""
        contributions: dict[str, float] = {}
        for feature, raw_key in _COUNTERFACTUAL_BINARY_FEATURES.items():
            current = int(raw[raw_key])
            alternative = 1 - current
            raw_alt = {**raw, raw_key: alternative}
            prob_current = float(self._model.predict_proba(pd.DataFrame([raw]))[0, 1])
            prob_alternative = float(self._model.predict_proba(pd.DataFrame([raw_alt]))[0, 1])
            contributions[feature] = prob_current - prob_alternative
        return contributions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clinical_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "muito_alto"
    if prob >= 0.50:
        return "alto"
    if prob >= DEFAULT_THRESHOLD:
        return "moderado"
    if prob >= _MODEL_PROB_CONCERN:
        return "zona_cinza"
    return "baixo"


def _clinical_note(feature: str, value: object, shap_value: float) -> str:
    direction = "aumentou" if shap_value > 0 else "reduziu"
    label = _FEATURE_LABELS.get(feature, feature)
    if value is None:
        return f"{label}: valor desconhecido — {direction} o risco estimado."
    return f"{label} = {value} → {direction} o risco estimado."


def _clinical_note_counterfactual(feature: str, value: object, delta: float) -> str:
    """Explain a binary flag relative to the opposite value on calibrated probability."""
    label = _FEATURE_LABELS.get(feature, feature)
    if value is None:
        direction = "aumentou" if delta > 0 else "reduziu"
        return f"{label}: valor desconhecido — {direction} o risco estimado."

    if feature == "PAI_AUSENTE":
        alt_desc = "presente na declaração (0)" if int(value) == 1 else "ausente na declaração (1)"
        if abs(delta) < 1e-6:
            return (
                f"{label} = {value} → efeito neutro na probabilidade calibrada "
                f"em relação a pai {alt_desc}."
            )
        direction = "aumentou" if delta > 0 else "reduziu"
        return (
            f"{label} = {value} → {direction} o risco estimado em relação a pai {alt_desc}."
        )

    direction = "aumentou" if delta > 0 else "reduziu"
    return f"{label} = {value} → {direction} o risco estimado."


def _top(fatores: list[FatorSHAP], positive: bool, n: int = 5) -> list[FatorSHAP]:
    subset = [f for f in fatores if (f.shap_value > 0) == positive]
    return sorted(subset, key=lambda f: abs(f.shap_value), reverse=True)[:n]
