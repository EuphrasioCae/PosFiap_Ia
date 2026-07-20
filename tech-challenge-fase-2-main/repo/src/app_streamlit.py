from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from src.llm_client import LLMError, generate_interpretation
from src.llm_context import build_llm_context
from src.predictor import PredictorPrematuro
from src.schemas import RequisicaoPredicao

load_dotenv(_ROOT / ".env")

ARTIFACTS_DIR = _ROOT / "results" / "artifacts"

CLINICAL_RISK_LABELS = {
    "muito_alto": "Muito alto",
    "alto": "Alto",
    "moderado": "Moderado",
    "zona_cinza": "Zona cinza",
    "baixo": "Baixo",
}

RISK_LABEL_PT = {
    "alto_risco_operacional": "Alto risco operacional",
    "baixo_risco": "Baixo risco",
}

ESCMAE2010_OPTIONS = {
    "0 — Analfabeta": 0.0,
    "1 — Fundamental I incompleto": 1.0,
    "2 — Fundamental I completo": 2.0,
    "3 — Fundamental II incompleto": 3.0,
    "4 — Fundamental II completo": 4.0,
    "5 — Superior completo": 5.0,
    "9 — Ignorado/desconhecido": 9.0,
}

KOTELCHUCK_OPTIONS = {
    "1 — Inadequado": "1",
    "2 — Intermediário I": "2",
    "3 — Intermediário II": "3",
    "4 — Adequado": "4",
    "5 — Adequado plus": "5",
    "9 — Ignorado/desconhecido": "9",
}


@st.cache_resource
def load_predictor() -> PredictorPrematuro:
    return PredictorPrematuro.from_artifacts_dir(ARTIFACTS_DIR)


def _factors_to_dataframe(factors) -> pd.DataFrame:
    if not factors:
        return pd.DataFrame(columns=["Fator", "Valor", "SHAP", "Nota clínica"])
    return pd.DataFrame(
        [
            {
                "Fator": f.label,
                "Valor": f.raw_value,
                "SHAP": round(f.shap_value, 4),
                "Nota clínica": f.clinical_note,
            }
            for f in factors
        ]
    )


def _render_form() -> dict | None:
    st.subheader("Dados da gestante")

    with st.form("dados_gestante", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            idade = st.number_input(
                "Idade da mãe (anos)",
                min_value=10,
                max_value=60,
                value=22,
            )
            esc_label = st.selectbox(
                "Escolaridade da mãe",
                options=list(ESCMAE2010_OPTIONS.keys()),
                index=2,
            )
            kotel_label = st.selectbox(
                "Adequação do pré-natal (Kotelchuck)",
                options=list(KOTELCHUCK_OPTIONS.keys()),
                index=2,
            )
            mes_prenat = st.number_input(
                "Mês de início do pré-natal (1–12; 99 = ignorado)",
                min_value=1,
                max_value=99,
                value=4,
            )
            qtd_gestant = st.number_input("Gestações anteriores", min_value=0, value=1)
            qtd_partnor = st.number_input("Partos normais", min_value=0, value=0)

        with col2:
            qtd_partces = st.number_input("Partos cesáreos", min_value=0, value=0)
            qtd_filvivo = st.number_input("Filhos vivos", min_value=0, value=1)
            qtd_filmort = st.number_input("Filhos mortos", min_value=0, value=0)
            latitude = st.number_input("Latitude", value=-19.9, format="%.4f")
            longitude = st.number_input("Longitude", value=-43.9, format="%.4f")
            pai_ausente = st.selectbox(
                "Pai ausente na declaração",
                options=["Não (0)", "Sim (1)"],
                index=0,
            )

        submitted = st.form_submit_button("Analisar risco", type="primary")

    if not submitted:
        return None

    return {
        "IDADEMAE": int(idade),
        "ESCMAE2010": ESCMAE2010_OPTIONS[esc_label],
        "KOTELCHUCK": KOTELCHUCK_OPTIONS[kotel_label],
        "MESPRENAT": int(mes_prenat),
        "QTDGESTANT": int(qtd_gestant),
        "QTDPARTNOR": int(qtd_partnor),
        "QTDPARTCES": int(qtd_partces),
        "QTDFILVIVO": int(qtd_filvivo),
        "QTDFILMORT": int(qtd_filmort),
        "LATITUDE": float(latitude),
        "LONGITUDE": float(longitude),
        "PAI_AUSENTE": 1 if pai_ausente.startswith("Sim") else 0,
    }


def _render_prediction(resp) -> None:
    st.divider()
    st.subheader("Resultado do modelo")

    col_prob, col_label, col_level = st.columns(3)
    col_prob.metric(
        "Probabilidade de prematuridade",
        f"{resp.risk_probability:.1%}",
    )
    col_label.metric(
        "Classificação operacional",
        RISK_LABEL_PT.get(resp.risk_label, resp.risk_label),
    )
    col_level.metric(
        "Nível de risco clínico",
        CLINICAL_RISK_LABELS.get(resp.clinical_risk_level, resp.clinical_risk_level),
    )

    st.caption(
        f"Threshold: {resp.threshold:.2f} | "
        f"Margem ao threshold: {resp.margin_to_threshold:+.4f}"
    )

    col_risk, col_prot = st.columns(2)
    df_risk = _factors_to_dataframe(resp.top_risk_factors)
    df_prot = _factors_to_dataframe(resp.top_protective_factors)

    with col_risk:
        st.markdown("**Fatores que aumentaram o risco (SHAP > 0)**")
        st.dataframe(
            df_risk,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SHAP": st.column_config.NumberColumn(format="%.4f"),
                "Nota clínica": st.column_config.TextColumn(width="large"),
            },
        )
    with col_prot:
        st.markdown("**Fatores que reduziram o risco (SHAP < 0)**")
        st.dataframe(
            df_prot,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SHAP": st.column_config.NumberColumn(format="%.4f"),
                "Nota clínica": st.column_config.TextColumn(width="large"),
            },
        )

    if df_risk.empty and df_prot.empty:
        st.info("Nenhum fator SHAP relevante identificado para este registro.")


def _render_interpretation(text: str) -> None:
    st.divider()
    st.subheader("Interpretação clínica (LLM)")
    st.markdown(text)
    st.caption(
        "Esta interpretação é gerada automaticamente e não substitui avaliação clínica."
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Sobre")
        st.markdown(
            "Ferramenta de apoio à decisão para estimativa de risco de parto prematuro, "
            "com explicabilidade SHAP e redação assistida por LLM."
        )
        mock = st.session_state.get("llm_mock", False)
        st.markdown(f"**Modo LLM:** {'Mock (offline)' if mock else 'OpenAI'}")
        st.markdown("**Modelo:** HistGradientBoosting calibrado (Fase 2)")
        st.markdown("**Threshold operacional:** 0,40")


def main() -> None:
    st.set_page_config(
        page_title="Risco de Prematuridade",
        layout="wide",
    )

    mock = os.getenv("LLM_MOCK", "0").strip() in {"1", "true", "True", "yes"}
    st.session_state["llm_mock"] = mock

    _render_sidebar()

    st.title("Risco de prematuridade")
    st.caption(
        "Modelo de apoio à decisão com explicabilidade SHAP e interpretação em linguagem natural."
    )

    payload = _render_form()
    if payload is None:
        st.info("Preencha o formulário e clique em **Analisar risco**.")
        return

    try:
        req = RequisicaoPredicao(**payload)
    except ValidationError as exc:
        st.error("Dados inválidos. Verifique os campos do formulário.")
        st.code(str(exc))
        return

    with st.spinner("Analisando risco e gerando interpretação..."):
        try:
            predictor = load_predictor()
            resp = predictor.predict(req)
            context = build_llm_context(resp)
            interpretation = generate_interpretation(context)
        except LLMError as exc:
            st.error(str(exc))
            return
        except FileNotFoundError as exc:
            st.error(
                "Artefatos do modelo não encontrados. "
                "Execute o notebook 07_training_pipeline.ipynb primeiro."
            )
            st.code(str(exc))
            return
        except Exception as exc:
            st.error(f"Erro inesperado durante a análise: {exc}")
            return

    _render_prediction(resp)
    _render_interpretation(interpretation)


if __name__ == "__main__":
    main()
