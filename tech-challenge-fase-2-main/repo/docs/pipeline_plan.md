# Plano — Pipeline único (treino + inferência)

Referência do time para consolidar os experimentos da Fase 1/2 num único
`sklearn.Pipeline` que **treina, é persistido e serve inferência** com o mesmo
pré-processamento. Hoje só existem os notebooks de experimento; não há pipeline
de produção.

## Princípio de desenho

O pré-processamento que depende dos dados (encode, scale) precisa estar **dentro**
do Pipeline para que treino e inferência usem exatamente a mesma transformação.
A limpeza do dataset histórico fica **fora** — em produção a entrada é validada
por contrato e dados inválidos/faltando viram erro na API, não imputação.

| Camada | O que faz | Roda em produção? |
|---|---|---|
| **Prep offline** | sentinela→NaN, plausibilidade, filtro `GRAVIDEZ==1`, dedup | Não — só prepara o dataset histórico de treino |
| **Pipeline sklearn** | feature engineering + one-hot + scaler + modelo | Sim — idêntico no treino e na inferência |
| **Serving / API** | validação de contrato → `predict_proba` → threshold | Sim |

> ⚠️ **Pré-requisito:** os parquet atuais (`X_train`/`X_test`) já vêm escalados e
> encodados. Para montar o pré-processamento dentro do Pipeline é preciso partir de
> um estágio anterior — o `df_model_raw.parquet` (ou um `df_clean` pré-encode), que
> está no repo da **Fase 1**. Trazer esse dado para a Fase 2 é o passo zero.

## Divisão: `src/` vs notebook

**`src/` = código reutilizável** (importado tanto pelo notebook de treino quanto pela API). É onde mora a lógica que precisa ser idêntica em treino e inferência. **Notebook = orquestração e demonstração** do treino. Só carrega dados, chama o `src/`, treina, avalia e persiste. Não contém regra de negócio.

**O que vai para `src/`**

- [ ] **`constants.py`** — listas de colunas, bins de idade, dicionário SINASC, colunas de leakage, threshold padrão (0.40).
- [ ] **`transformers.py`** — `FeatureEngineer` (`BaseEstimator`/`TransformerMixin`, stateless): `PNTARDIO`, `HISTPERDAFETAL`, `PRIMIPARA`, `PAI_AUSENTE`, `FAIXAETAMAE`, `ESCMAE2010_ORDINAL`, `KOTELCHUCK_ORDINAL`.
- [ ] **`pipeline_factory.py`** — `build_pipeline()` que monta (sem fit) o `ColumnTransformer` (`OneHotEncoder(handle_unknown="ignore")` nas categóricas + `StandardScaler` nas contínuas/ordinais + passthrough nas flags) + `HistGradientBoostingClassifier` com os hiperparâmetros do experimento 03.
- [ ] **`cleaning.py`** — limpeza offline do dataset histórico (sentinela, plausibilidade, escopo, dedup). Usada só no treino, mas isolada do notebook.
- [ ] **`schema.py`** — contrato de entrada (Pydantic) + validações de domínio/faixa usadas pela API.
- [ ] **`serving.py`** — `load_model()`, `predict(payload)`: valida → `predict_proba` → aplica threshold → resposta.
- [ ] **`explainability.py`** — geração dos sinais interpretáveis por registro (SHAP local/waterfall em formato tabular, top fatores pró/contra risco, importância global e metadados do modelo) para enriquecer a saída da LLM.
- [ ] **`llm_context.py`** — monta um JSON enxuto, estável e seguro para prompt: decisão do modelo + explicações + alertas de leitura clínica, sem delegar cálculo ou interpretação numérica para a LLM.

**O que vai no notebook**

- [x] Trazer `df_model_raw.parquet` da Fase 1 para `data/`.
- [ ] `from src.cleaning import ...` → gerar `df_clean`.
- [ ] Criar alvo `PREMATURO = SEMAGESTAC < 37`, dropar colunas de leakage, `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
- [ ] `pipeline = build_pipeline()` → `pipeline.fit(X_train, y_train, model__sample_weight=balanced)`.
- [ ] Avaliar no teste @ threshold 0.40 (recall, F2, matriz de confusão) e conferir que bate com os números do experimento 03.
- [ ] `joblib.dump(pipeline, ...)` — Pipeline inteiro (pré-processamento + modelo).
- [ ] Gerar e persistir artefatos de interpretabilidade: `shap_background.parquet` (amostra de treino transformada), `global_feature_importance.parquet` (`mean_abs_shap`) e `feature_descriptions.json` (rótulos clínicos/negócio por feature).
- [ ] Smoke test: `from src.serving import predict`, passar 1 registro cru válido.

## Threshold

O modelo retorna **probabilidade** (`predict_proba`), não a classe. O *threshold* é o ponto de corte acima do qual o caso é rotulado como prematuro.

- **Não está dentro do modelo:** `pipeline.predict()` usa 0.5 fixo e escondido. Para usar 0.40 é preciso chamar `predict_proba` e aplicar o corte na camada de serviço (`src/serving.py`). É uma decisão **operacional**, separada do treino — muda sem retreinar.
- **Controla o trade-off recall × precisão:**
  - corte **baixo** (0.40) → marca mais → **recall ↑** (pega mais prematuro), FP ↑.
  - corte **alto** (0.50) → marca menos → FN ↑ (deixa passar prematuro — erro grave).
- **Decisão fixada:** threshold **0.40 com piso clínico de recall ≥ 0.80** — priorizar não deixar prematuro passar, aceitando mais falsos positivos. O experimento 06 confirmou 0.40 como ótimo para o cenário B.
- **Fixo vs. parâmetro (v1):** começar **fixo em 0.40** (constante em `src/constants.py`, garante o piso de recall). Expor como parâmetro fica como evolução futura, se surgir necessidade (ex. unidade de alto risco com corte menor).

## Enriquecimento para a LLM

A inferência não deve devolver só `probability` e `prediction`. A API precisa montar um **contexto explicável** para a LLM gerar uma resposta útil, auditável e alinhada ao modelo. A referência principal é `notebooks/04_interpretability.ipynb` da Fase 1, especialmente a seção de SHAP local/waterfall: ela decompõe a previsão de um registro em contribuições por feature.

### Quais dados enviar

| Dado | Origem | Uso pela LLM |
|---|---|---|
| `risk_probability` | `pipeline.predict_proba(payload)[1]` | comunicar risco estimado |
| `threshold` | `DEFAULT_THRESHOLD = 0.40` | explicar a regra de decisão |
| `risk_label` | `risk_probability >= threshold` | classificar como alto/baixo risco operacional |
| `margin_to_threshold` | `risk_probability - threshold` | indicar quão perto o caso está do ponto de corte |
| `top_risk_factors` | maiores SHAP positivos do registro | listar fatores que aumentaram o risco |
| `top_protective_factors` | maiores SHAP negativos do registro | listar fatores que reduziram o risco |
| `feature_values` | payload bruto + features derivadas | traduzir fatores para valores entendíveis |
| `global_feature_rank` | `mean_abs_shap` salvo no treino | contextualizar se o fator é estruturalmente relevante |
| `clinical_notes` | `feature_descriptions.json` curado | transformar feature técnica em linguagem clínica |
| `interpretation_warnings` | constantes curadas | lembrar que SHAP é associação, não causalidade |
| `model_metadata` | artefato salvo no treino | versão, data, threshold, métricas e cenário usado |

### Como pegar esses dados

1. **Na inferência**, validar o payload com `schema.py` e manter uma cópia do registro bruto validado para exibição.
2. Chamar `pipeline.predict_proba` e aplicar o threshold 0.40 em `serving.py`.
3. Transformar o mesmo payload com o pré-processador do Pipeline (`pipeline[:-1].transform(...)`) para obter a matriz que o modelo realmente vê.
4. Recuperar nomes finais das features com `get_feature_names_out()` do `ColumnTransformer`; para features derivadas, usar nomes definidos em `FeatureEngineer`.
5. Calcular SHAP local em `explainability.py`:
   - para `HistGradientBoostingClassifier`, usar `shap.TreeExplainer(model)` no registro transformado;
   - se o modelo final estiver calibrado (`CalibratedClassifierCV`), repetir a lógica do notebook da Fase 1: calcular SHAP nos estimadores internos e usar a média das contribuições e dos `base_values`.
6. Converter o resultado SHAP em tabela ordenada: `feature`, `raw_value`, `transformed_value`, `shap_value`, `direction` (`risk`/`protective`), `abs_rank`, `global_rank`, `label`, `clinical_note`.
7. Selecionar os **top 5 fatores de risco** (`shap_value > 0`) e os **top 5 fatores protetores** (`shap_value < 0`) para caber no prompt sem poluir a LLM.
8. Montar o `llm_context` como JSON determinístico; a LLM deve **redigir** a resposta, não recalcular probabilidade, threshold, ranking ou sinais de risco.

### Exemplo de contexto para prompt

```json
{
  "risk_probability": 0.67,
  "threshold": 0.4,
  "risk_label": "alto_risco_operacional",
  "margin_to_threshold": 0.27,
  "top_risk_factors": [
    {
      "feature": "KOTELCHUCK_ORDINAL",
      "label": "adequação do pré-natal",
      "raw_value": "Inadequado",
      "shap_value": 0.31,
      "clinical_note": "pré-natal inadequado aumentou o risco estimado"
    },
    {
      "feature": "MESPRENAT",
      "label": "mês de início do pré-natal",
      "raw_value": 5,
      "shap_value": 0.14,
      "clinical_note": "início tardio reduziu a janela de acompanhamento"
    }
  ],
  "top_protective_factors": [
    {
      "feature": "QTDFILVIVO",
      "label": "filhos vivos anteriores",
      "raw_value": 2,
      "shap_value": -0.08,
      "clinical_note": "histórico obstétrico prévio atuou como fator protetor no modelo"
    }
  ],
  "interpretation_warnings": [
    "SHAP mede associação do modelo, não causalidade clínica.",
    "KOTELCHUCK pode sofrer causalidade reversa em nascimentos prematuros.",
    "A resposta não substitui avaliação profissional."
  ]
}
```

### Features candidatas para descrição curada

Do notebook de interpretabilidade da Fase 1, começar pelas features com maior valor explicativo global:

| Feature | Leitura para a LLM |
|---|---|
| `KOTELCHUCK_ORDINAL` | adequação do pré-natal; maior inadequação tende a aumentar o risco |
| `MESPRENAT` | mês de início do pré-natal; início tardio tende a aumentar o risco |
| `IDADEMAE` | idade materna; extremos de idade podem aumentar risco no padrão em U |
| `QTDFILVIVO` | histórico de filhos vivos; pode atuar como proxy protetor de gestações anteriores |
| `LONGITUDE` | localização/região; proxy de desigualdade territorial e acesso à saúde |

> Guardrail: a LLM não deve afirmar causa direta. Para `KOTELCHUCK_ORDINAL`, incluir
> sempre o alerta de que parte do sinal pode refletir causalidade reversa, como já
> documentado no notebook da Fase 1.

## Decisões em aberto

1. **Features de ausência** (`KOTELCHUCK_IGNORADO`, `ESCMAE2010_IGNORADO`, convenções `-1`): se o contrato não admite dado faltando, elas ficam constantes. Como é um retreino do zero, a recomendação é **removê-las** para um contrato limpo.
2. **Hiperparâmetros do experimento 03:** confirmar/extrair os valores finais para instanciar o `HistGradientBoostingClassifier`.
3. **Custo de SHAP em produção:** decidir se SHAP local roda síncrono em toda chamada ou se começa atrás de flag (`include_explanation=true`) para controlar latência.
