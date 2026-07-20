from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.constants import (
    DEFAULT_THRESHOLD,
    IDADEMAE_MAX,
    IDADEMAE_MIN,
    MESPRENAT_MAX,
    MESPRENAT_MIN,
)

# ---------------------------------------------------------------------------
# Contrato de ENTRADA
# Campos do Cenário B (SINASC/MG, 2020–2022). Valores sentinela (9, 99, "9") são aceitos —
# a conversão sentinela → NaN é feita pelo cleaning.py no treino; em produção
# o pipeline sklearn lida nativamente com NaN via HistGradientBoosting.
# ---------------------------------------------------------------------------


class RequisicaoPredicao(BaseModel):
    """Contrato de entrada para o endpoint de predição de prematuridade."""

    # --- dados demográficos ---
    IDADEMAE: int = Field(
        ge=IDADEMAE_MIN,
        le=IDADEMAE_MAX,
        description=f"Idade da mãe em anos ({IDADEMAE_MIN}–{IDADEMAE_MAX})",
    )

    # --- escolaridade ---
    ESCMAE2010: float = Field(
        description="Escolaridade da mãe (0–5); use 9 para ignorado/desconhecido"
    )

    # --- índice de Kotelchuck ---
    KOTELCHUCK: str = Field(
        description=(
            "Adequação do pré-natal pelo índice de Kotelchuck ('1'–'5'); "
            "use '9' para ignorado/desconhecido"
        )
    )

    # --- gestação atual ---
    MESPRENAT: int = Field(
        description=(
            f"Mês de início do pré-natal ({MESPRENAT_MIN}–{MESPRENAT_MAX}); "
            "use 99 para ignorado/desconhecido"
        )
    )

    # --- histórico obstétrico ---
    QTDGESTANT: int = Field(ge=0, description="Número de gestações anteriores")
    QTDPARTNOR: int = Field(ge=0, description="Número de partos normais")
    QTDPARTCES: int = Field(ge=0, description="Número de partos cesáreos")
    QTDFILVIVO: int = Field(ge=0, description="Número de filhos vivos")
    QTDFILMORT: int = Field(ge=0, description="Número de filhos mortos")

    # --- geolocalização ---
    LATITUDE: float = Field(description="Latitude do local de nascimento")
    LONGITUDE: float = Field(description="Longitude do local de nascimento")

    # --- flag derivada (da declaração de nascimento) ---
    PAI_AUSENTE: Literal[0, 1] = Field(
        description="Pai ausente na declaração de nascimento (0=presente, 1=ausente)"
    )

    # ------------------------------------------------------------------
    # Validação estrutural — garante valores possíveis do SINASC.
    # Sentinelas são aceitos; a conversão para NaN fica no pipeline.
    # ------------------------------------------------------------------

    @field_validator("MESPRENAT")
    @classmethod
    def _valida_mesprenat(cls, v):
        sentinela = 99
        if v != sentinela and not (MESPRENAT_MIN <= v <= MESPRENAT_MAX):
            raise ValueError(
                f"MESPRENAT deve ser {MESPRENAT_MIN}–{MESPRENAT_MAX} ou 99 (ignorado), "
                f"recebido: {v}"
            )
        return v

    @field_validator("ESCMAE2010")
    @classmethod
    def _valida_escmae2010(cls, v):
        valores_validos = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0}
        if v not in valores_validos:
            raise ValueError(
                f"ESCMAE2010 deve ser um de {sorted(valores_validos)} (9=ignorado), "
                f"recebido: {v}"
            )
        return v

    @field_validator("KOTELCHUCK")
    @classmethod
    def _valida_kotelchuck(cls, v):
        valores_validos = {"1", "2", "3", "4", "5", "9"}
        if v not in valores_validos:
            raise ValueError(
                f"KOTELCHUCK deve ser um de {sorted(valores_validos)} ('9'=ignorado), "
                f"recebido: {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Contrato de SAÍDA
# ---------------------------------------------------------------------------


class FatorSHAP(BaseModel):
    """Um fator de risco ou proteção identificado via SHAP local."""

    feature: str = Field(description="Nome técnico da feature")
    label: str = Field(description="Rótulo clínico legível")
    raw_value: Any = Field(description="Valor bruto enviado na requisição")
    shap_value: float = Field(description="Contribuição SHAP (positivo = risco, negativo = proteção)")
    clinical_note: str = Field(description="Interpretação clínica deste fator para este registro")


class RespostaPredicao(BaseModel):
    """Contrato de saída do endpoint de predição de prematuridade.

    Estruturado para alimentar diretamente a LLM via llm_context.py —
    a LLM redige a resposta com base nesses campos, sem recalcular nada.
    """

    risk_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probabilidade de prematuridade estimada pelo modelo",
    )
    risk_label: Literal["alto_risco_operacional", "baixo_risco"] = Field(
        description=f"Classificação operacional (corte: threshold={DEFAULT_THRESHOLD})"
    )
    clinical_risk_level: Literal[
        "muito_alto", "alto", "moderado", "zona_cinza", "baixo"
    ] = Field(
        description=(
            "Nível de risco clínico relativo à escala real do modelo. "
            "muito_alto: prob≥0.70 (top 10% dos prematuros reais). "
            "alto: 0.50≤prob<0.70. "
            "moderado: threshold≤prob<0.50. "
            "zona_cinza: 0.35≤prob<threshold (12% dos FN caem aqui — risco não nulo). "
            "baixo: prob<0.35 (abaixo do p5 dos prematuros reais)."
        )
    )
    threshold: float = Field(
        description="Limiar de decisão utilizado"
    )
    margin_to_threshold: float = Field(
        description="Distância da probabilidade ao threshold (positivo = acima do corte)"
    )
    top_risk_factors: list[FatorSHAP] | None = Field(
        default=None,
        description="Top 5 fatores que aumentaram o risco (SHAP > 0), ordenados por |shap_value|",
    )
    top_protective_factors: list[FatorSHAP] | None = Field(
        default=None,
        description="Top 5 fatores que reduziram o risco (SHAP < 0), ordenados por |shap_value|",
    )
    interpretation_warnings: list[str] | None = Field(
        default=None,
        description="Alertas de leitura clínica a serem incluídos sempre pela LLM",
    )
