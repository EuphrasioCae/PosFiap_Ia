# Issues — Sprint pipeline + interpretabilidade

Dois tracks paralelos. Ponto de sincronização: **artefato do modelo treinado** (A5) — desbloqueia B2 em diante.

```
Track A — Caê (treino)                 Track B — Emídio (inferência + interpretabilidade)
──────────────────────────────         ──────────────────────────────────────────────────
A1 constants.py                        B1 feature_descriptions.json  ← pode começar agora
    ↓                                  B2 schema.py                  ← pode começar após A1
A2 cleaning.py                              ↓
A3 transformers.py                     (aguarda A5)
    ↓                                       ↓
A4 pipeline_factory.py                 B3 explainability.py
    ↓                                       ↓
A5 notebook de treino ─────────────→   B4 artefatos no notebook
                                            ↓
                       A1 + B2 ────→   B5 serving.py
                       A5 + B3 ────→   B6 llm_context.py
```

---

## Bloqueadores a resolver antes de começar

| # | Decisão | Impacto |
|---|---|---|
| D1 | Re-rodar `03_AG_expanded_search.ipynb` para extrair `best_params` do **Exp1_conservador** — os outputs foram removidos pelo nbstripout no commit de hoje. Seed fixo (42), resultado reproduzível. Modelo escolhido: 03 Exp1, melhor em redução absoluta de FN (+167 prematuros detectados, −6.2% FN) | bloqueia A4 |
| D2 | Remover `KOTELCHUCK_IGNORADO` / `ESCMAE2010_IGNORADO` do contrato (recomendação: sim, retreino do zero) | bloqueia A1, A3, B2 |
| D3 | SHAP síncrono em toda chamada ou atrás de flag `include_explanation=true` | bloqueia B3, B5 |

**Referência dos params do baseline** (Fase 1 — ponto de partida do AG 03): `max_iter=350`, `learning_rate=0.05`, `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=1.0`, `max_depth=None`, `max_bins=255`.

---

## Track A — Caê

### A1 · `src/constants.py`

**Depende de:** D2

- Listas de colunas: `RAW_COLUMNS`, `LEAKAGE_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`, `PASSTHROUGH_COLUMNS`
- Bins e labels para `FAIXAETAMAE`
- `KOTELCHUCK_MAP` e `ESCMAE2010_MAP` (ordinais)
- `DEFAULT_THRESHOLD = 0.40`, `RANDOM_STATE = 42`, `SEMAGESTAC_PRETERM_CUTOFF = 37`

**Critério de aceite:** `from src.constants import DEFAULT_THRESHOLD` importa sem erro; todas as listas têm type hint.

---

### A2 · `src/cleaning.py`

**Depende de:** A1

Limpeza offline do `df_model_raw.parquet` — roda só no treino.

- Filtrar `GRAVIDEZ == 1` (excluir gemelares: 17k registros confirmados)
- Sentinela → NaN (`MESPRENAT=99`, `ESCMAE2010=9`, etc.)
- Validações de plausibilidade (`IDADEMAE` fora de [10, 60], etc.)
- Deduplicação
- Dropar colunas fora do escopo: `SEMAGESTAC`, `GESTACAO`, `ESTCIVMAE`, `RACACORMAE`, `SEXO`, `CONSPRENAT`, `IDADEPAI`, `IDADEPAI_INVALIDA`

**Critério de aceite:** `clean_dataset(df_raw)` retorna DataFrame com `GRAVIDEZ` removido; shape esperado ~700k linhas.

---

### A3 · `src/transformers.py`

**Depende de:** A1

`FeatureEngineer(BaseEstimator, TransformerMixin)` — stateless (`.fit()` não salva estado).

Features derivadas:
- `PNTARDIO` — `MESPRENAT > 4`
- `HISTPERDAFETAL` — `QTDFILMORT > 0`
- `PRIMIPARA` — `QTDPARTNOR + QTDPARTCES == 0`
- `FAIXAETAMAE` — bins de `IDADEMAE` via `constants.py`
- `ESCMAE2010_ORDINAL` — mapeamento via `ESCMAE2010_MAP`
- `KOTELCHUCK_ORDINAL` — mapeamento via `KOTELCHUCK_MAP`

**Critério de aceite:** `FeatureEngineer().fit_transform(X)` idempotente; funciona dentro de `sklearn.Pipeline`; `get_feature_names_out()` retorna nomes corretos.

---

### A4 · `src/pipeline_factory.py`

**Depende de:** A1, A3, D1

`build_pipeline() -> sklearn.Pipeline` — monta sem fit.

1. `FeatureEngineer`
2. `ColumnTransformer`: `OneHotEncoder(handle_unknown="ignore")` nas categóricas + `StandardScaler` nas numéricas/ordinais + `passthrough` nas flags
3. `HistGradientBoostingClassifier(**BEST_PARAMS)` — params do **03 Exp1_conservador** (ver D1)

**Critério de aceite:** `pipe = build_pipeline(); pipe.fit(X_train, y_train)` sem erro; `pipe[1].get_feature_names_out()` retorna nomes das features finais.

---

### A5 · Notebook `07_training_pipeline.ipynb`

**Depende de:** A2, A4

Sem lógica de negócio — só orquestra o `src/`.

1. `cleaning.clean_dataset(df_raw)` → `df_clean`
2. Criar `PREMATURO`, dropar `SEMAGESTAC` e leakage, `train_test_split(0.2, stratify=y, random_state=42)`
3. `pipeline.fit(X_train, y_train, histgradientboostingclassifier__sample_weight=balanced)`
4. Avaliar no teste @ 0.40: recall ≥ 0.80, comparar com exp 03
5. `joblib.dump(pipeline, "results/artifacts/pipeline_v2.pkl")`
6. Persistir `model_metadata.json` (data, threshold, métricas, cenário, hiperparâmetros)

**Critério de aceite:** `pipeline_v2.pkl` existe; recall no teste ≥ 0.80 @ 0.40.

---

## Track B — Emídio

### B1 · `data/feature_descriptions.json`

**Depende de:** nada — pode começar agora

Dicionário curado com rótulo clínico por feature.

Começar pelas features com maior SHAP global (Fase 1, `04_interpretability.ipynb`):

| Feature | Rótulo sugerido |
|---|---|
| `KOTELCHUCK_ORDINAL` | adequação do pré-natal |
| `MESPRENAT` | mês de início do pré-natal |
| `IDADEMAE` | idade materna |
| `QTDFILVIVO` | filhos vivos anteriores |
| `LONGITUDE` | localização / acesso à saúde |
| `PNTARDIO` | início tardio do pré-natal |
| `PRIMIPARA` | primeira gestação |
| `HISTPERDAFETAL` | histórico de perda fetal |
| `ESCMAE2010_ORDINAL` | escolaridade materna |

`KOTELCHUCK_ORDINAL` deve ter `causal_warning` com o alerta de causalidade reversa documentado na Fase 1.

**Critério de aceite:** JSON válido; todas as features do cenário B têm entrada com `label`, `clinical_note_risk`, `clinical_note_protective`, `causal_warning` (nullable).

---

### B2 · `src/schema.py`

**Depende de:** A1

Contrato de entrada (Pydantic). Campos do cenário B com tipo, faixa válida e `description` clínico:

`IDADEMAE`, `ESCMAE2010`, `KOTELCHUCK`, `MESPRENAT`, `QTDGESTANT`, `QTDPARTNOR`, `QTDPARTCES`, `QTDFILVIVO`, `QTDFILMORT`, `LATITUDE`, `LONGITUDE`

**Critério de aceite:** payload inválido levanta `ValidationError`; payload válido serializa para `pd.DataFrame` de 1 linha compatível com `pipeline.predict_proba`.

---

### B3 · `src/explainability.py`

**Depende de:** A5 (artefato), D3

- `explain_local(pipeline, X_raw_1row, feature_descriptions) -> list[FactorRow]`
  - `pipeline[:-1].transform(X_raw_1row)` → matriz que o modelo vê
  - `shap.TreeExplainer(pipeline[-1])` no registro transformado
  - Se `CalibratedClassifierCV`: SHAP nos estimadores internos, média dos `base_values` (padrão `04_interpretability.ipynb` Fase 1)
  - Retorna tabela: `feature`, `raw_value`, `shap_value`, `direction`, `abs_rank`, `global_rank`, `label`, `clinical_note`
- `get_global_importance(pipeline, X_background) -> pd.DataFrame` — `mean_abs_shap` por feature

**Critério de aceite:** `explain_local` retorna lista ordenada por `abs(shap_value)` desc; top 5 risco + top 5 proteção corretos; testado com o registro de exemplo do `pipeline_plan.md`.

---

### B4 · Artefatos de interpretabilidade

**Depende de:** B3, A5

Adicionar ao notebook `07` (ou `08_explainability_artifacts.ipynb`):

- `results/artifacts/shap_background.parquet` — 500–1k registros de treino já transformados pelo pipeline
- `results/artifacts/global_feature_importance.parquet` — `mean_abs_shap` por feature, ordenado desc
- `results/artifacts/feature_descriptions.json` — cópia de B1 persistida junto ao modelo

**Critério de aceite:** três arquivos existem em `results/artifacts/`; `global_feature_importance.parquet` tem colunas `feature` e `mean_abs_shap`.

---

### B5 · `src/serving.py`

**Depende de:** A1, A5 (artefato), B2, D3

- `load_model(path) -> Pipeline`
- `predict(payload: dict, include_explanation: bool = False) -> PredictionResult`
  - Valida via `schema.py` → `predict_proba` → aplica threshold
  - Se `include_explanation=True`: chama `explainability.explain_local`

**Critério de aceite:** smoke test com registro válido retorna `risk_probability`, `risk_label`, `threshold`; registro inválido levanta erro antes de chegar no modelo.

---

### B6 · `src/llm_context.py`

**Depende de:** B3, B2

`build_llm_context(prediction_result, explanation, feature_descriptions, model_metadata) -> dict`

Monta JSON determinístico (ver exemplo em `pipeline_plan.md`):
- `risk_probability`, `threshold`, `risk_label`, `margin_to_threshold`
- `top_risk_factors` (top 5, `shap_value > 0`)
- `top_protective_factors` (top 5, `shap_value < 0`)
- `interpretation_warnings` (constantes; sempre incluir SHAP ≠ causalidade)
- `model_metadata`

A LLM recebe esse JSON e **redige** — não recalcula nada.

**Critério de aceite:** saída é JSON serializável; máximo 5 fatores por lado; sem campos que a LLM poderia tentar recalcular.

---

## Resumo

| Issue | Responsável | Pode começar? | Aguarda |
|---|---|---|---|
| A1 · constants.py | Caê | Após D2 | D2 |
| A2 · cleaning.py | Caê | Após A1 | A1 |
| A3 · transformers.py | Caê | Após A1 | A1 |
| A4 · pipeline_factory.py | Caê | Após A1, A3, D1 | A1, A3, D1 |
| A5 · notebook de treino | Caê | Após A2, A4 | A2, A4 |
| B1 · feature_descriptions.json | Emídio | **Agora** | — |
| B2 · schema.py | Emídio | Após A1 | A1 |
| B3 · explainability.py | Emídio | Após A5 | A5 |
| B4 · artefatos | Emídio | Após B3, A5 | B3, A5 |
| B5 · serving.py | Emídio | Após A1, A5, B2, D3 | A5, B2, D3 |
| B6 · llm_context.py | Emídio | Após B3, B2 | B3, B2 |
