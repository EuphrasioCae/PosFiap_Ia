# Tech Challenge Fase 2

FIAP Pós Tech IA para Devs. Continuação do projeto do hospital universitário iniciado na Fase 1.

## Desafio

Após o sucesso inicial no desenvolvimento de modelos de machine learning para diagnóstico médico no Módulo 1, o hospital universitário agora enfrenta novos desafios que podem ser solucionados com técnicas de algoritmos genéticos e processamento de linguagem natural.

## Projeto escolhido: Otimização de Modelos de Diagnóstico

O hospital precisa melhorar a precisão e eficiência dos modelos de diagnóstico desenvolvidos no Módulo 1. O desafio é utilizar algoritmos genéticos para otimizar os hiperparâmetros desses modelos, além de incorporar capacidades iniciais de processamento de linguagem natural através de LLMs para melhorar a interpretabilidade dos resultados para os profissionais de saúde.

Esta fase é fundamental para preparar a infraestrutura necessária para o assistente médico mais avançado previsto no Módulo 3.

## Objetivo

Desenvolver uma solução para otimização do modelo de ML médico existente (HistGradientBoostingClassifier da Fase 1), além de implementar recursos iniciais de processamento de linguagem natural para melhorar a interpretação e apresentação dos diagnósticos.

## Requisitos obrigatórios

### 1. Otimização via Algoritmos Genéticos

* Implementar um algoritmo genético para otimização de hiperparâmetros do modelo de diagnóstico desenvolvido no Módulo 1:
  * Definir uma codificação adequada (representação de genes) para os hiperparâmetros relevantes.
  * Implementar operadores de seleção, cruzamento e mutação.
  * Definir uma função fitness baseada nas métricas de desempenho do modelo (accuracy, recall, F1-score, etc.).
* Comparar o desempenho do modelo otimizado com o modelo original da Fase 1.
* Realizar ao menos 3 experimentos com diferentes configurações do algoritmo genético (tamanho da população, taxas de mutação, etc.).

### 2. Escalabilidade automática

* Configurar recursos de escalabilidade automática para lidar com variações de demanda.
* Implementar monitoramento e logging adequados para tracking de desempenho.
* Documentar arquitetura e decisões de implementação.

Observação: a implementação em nuvem é opcional e pode ser considerada para pontuação extra.

### 3. Integração com LLMs para interpretação de resultados

* Integrar uma LLM pré-treinada (GPT, Falcon, LLaMA, etc.) para:
  * Gerar explicações em linguagem natural dos diagnósticos produzidos pelo modelo.
  * Transformar dados numéricos e estatísticos em insights acionáveis para médicos.
  * Preparar a base para a futura integração com dados textuais no Módulo 3.
* Implementar técnicas de prompt engineering para obter respostas relevantes e adequadas ao contexto médico.
* Avaliar a qualidade das interpretações geradas.

### 4. Código e organização

* Projeto Python bem estruturado, utilizando ambiente virtual (este repositório usa `uv`).
* Documentação detalhada, incluindo diagramas de arquitetura.
* Testes automatizados para validação de funcionalidades.
* Se optado pela implementação em nuvem: Infraestrutura como código (IaC) para provisionamento dos recursos.

## Etapas e responsáveis

| Etapa | Responsável | Janela |
|---|---|---|
| 1. AG para otimização de hiperparâmetros | Isa | 08/06 a 14/06 |
| 1.5. Contrato de dados modelo para LLM e spec de logging | Emídio e Caê | 15/06 a 21/06 |
| 2. Escalabilidade, monitoramento e logging | Alan | 15/06 a 24/06 |
| 3. Integração com LLM para interpretação | Igor | 22/06 a 30/06 |
| 4. Integração end to end e testes automatizados | Todos | 01/07 a 06/07 |
| 5. Relatório técnico e vídeo | Emídio e Caê | 07/07 a 13/07 |

Entrega final: 14/07.

## Entregáveis

* **Repositório Git** com código-fonte completo, documentação da API (se aplicável), scripts e notebooks de demonstração, e arquivos de configuração para implantação (se em nuvem).
* **Relatório técnico** cobrindo: implementação do AG e resultados da otimização de hiperparâmetros, integração com LLM (abordagem, prompts utilizados, avaliação de qualidade), comparativo de desempenho entre modelo original e otimizado, desafios enfrentados e arquitetura da solução.
* **Vídeo de demonstração** de até 15 minutos no YouTube ou Vimeo (público ou não listado), com demonstração do sistema em execução, explicação dos componentes, apresentação dos resultados do AG e demonstração da integração com LLM.

## Estrutura do repositório

```
.
├── data/                          splits prontos (X/y train/test) gerados na Fase 1
├── notebooks/
│   ├── 01_AG.ipynb                    AG principal (recall e F2)
│   ├── 02_AG_threshold040.ipynb       AG com threshold operacional 0.40
│   ├── 03_AG_expanded_search.ipynb    AG com espaço expandido (8 genes)
│   ├── 04_AG_smote.ipynb              AG com SMOTE para balanceamento
│   ├── 05_AG_random_forest.ipynb      AG sobre RandomForest (alternativa ao HGB)
│   ├── 06_co_evolution.ipynb          Co-evolução de hiperparâmetros e threshold
│   ├── 07_model_selection.ipynb       Seleção calibrada do modelo final (comparativo multi-cenário)
│   └── 07_training_pipeline.ipynb     Treino final + SHAP + exportação de artefatos
├── results/
│   ├── AG_resultados_resumo.md        síntese consolidada dos experimentos
│   ├── artifacts/                     modelos treinados e bundle SHAP
│   │   ├── model_phase2.pkl           modelo calibrado de produção (Fase 2)
│   │   ├── shap_bundle.pkl            explainer + preprocessor + feature_names
│   │   └── shap_values_sample.npy     SHAP values de 2k amostras do teste
│   ├── figures/                       gráficos de convergência e SHAP
│   ├── metrics/                       métricas dos experimentos e do modelo final
│   └── params/                        best_params JSON de cada experimento AG
├── src/                           pacote Python instalável (raiz: src/, via pyproject.toml)
│   ├── cleaning.py                limpeza e remoção de leakage do dataset bruto
│   ├── constants.py               hiperparâmetros, colunas e limiares operacionais
│   ├── pipeline_factory.py        build_pipeline() — pipeline sklearn de produção
│   ├── predictor.py               PredictorPrematuro — classe de inferência com SHAP
│   ├── schemas.py                 contratos Pydantic de entrada/saída (RequisicaoPredicao, RespostaPredicao)
│   ├── transformers.py            FeatureEngineer (transformer sklearn customizado)
│   └── utils/                     subpacote `utils` — importável nos notebooks sem sys.path
│       ├── __init__.py
│       └── experiment_utils.py    helpers de modelagem sklearn + I/O de notebooks
├── docs/
│   ├── pipeline_plan.md           plano do pipeline de produção (treino + inferência)
│   └── issues_sprint.md           organização das issues para Emídio e Caê
└── pyproject.toml
```

## Setup

```bash
uv sync
uv run jupyter lab
```

### Configuração do nbstripout (obrigatório)

O projeto usa [nbstripout](https://github.com/kynan/nbstripout) para evitar que outputs e metadados dos notebooks sejam commitados. **Todos os membros do time precisam rodar este comando uma vez após clonar o repositório:**

```bash
uv run nbstripout --install --attributes .gitattributes
```

Isso registra o filtro do Git localmente. O arquivo `.gitattributes` já está no repositório com as regras:

```gitattributes
*.ipynb filter=nbstripout
*.zpln filter=nbstripout
*.ipynb diff=ipynb
```

Verifique se está ativo com `uv run nbstripout --status`.

## Modelo de produção e inferência

### Treinar e salvar o modelo

Execute o notebook `notebooks/07_training_pipeline.ipynb` do início ao fim. Ele:

1. Carrega `data/df_model_raw.parquet` via `clean_dataset()`
2. Treina o pipeline final (hiperparâmetros do AG 03/Exp2_padrao, calibração isotônica cv=3)
3. Computa o SHAP `TreeExplainer` sobre o HistGB base
4. Salva três artefatos em `results/artifacts/`:

| Arquivo | Conteúdo |
|---|---|
| `model_phase2.pkl` | `CalibratedClassifierCV` completo — aceita DataFrame bruto diretamente |
| `shap_bundle.pkl` | `{'explainer', 'preprocessor', 'feature_names'}` para SHAP por predição |
| `shap_values_sample.npy` | SHAP values de 2 000 amostras do teste (validação e visualização) |

### Usar o preditor em código (Streamlit, FastAPI, etc.)

```python
from src.predictor import PredictorPrematuro
from src.schemas import RequisicaoPredicao, RespostaPredicao

# Carrega uma vez; use @st.cache_resource no Streamlit
predictor = PredictorPrematuro.from_artifacts_dir("results/artifacts")

req = RequisicaoPredicao(
    IDADEMAE=22,
    ESCMAE2010=2.0,       # 0–5; use 9 para ignorado
    KOTELCHUCK="3",       # "1"–"5"; use "9" para ignorado
    MESPRENAT=4,          # 1–12; use 99 para ignorado
    QTDGESTANT=1,
    QTDPARTNOR=0,
    QTDPARTCES=0,
    QTDFILVIVO=1,
    QTDFILMORT=0,
    LATITUDE=-19.9,
    LONGITUDE=-43.9,
    PAI_AUSENTE=0,        # 0 = presente, 1 = ausente
)

resp: RespostaPredicao = predictor.predict(req)

print(resp.risk_probability)       # ex: 0.4918
print(resp.risk_label)             # "alto_risco_operacional" | "baixo_risco"
print(resp.clinical_risk_level)    # "muito_alto" | "alto" | "moderado" | "zona_cinza" | "baixo"
print(resp.top_risk_factors)       # lista de FatorSHAP (feature, label, shap_value, clinical_note)
print(resp.interpretation_warnings)# contexto de escala para a LLM
```

**Campos de `RespostaPredicao` relevantes para a LLM:**

| Campo | Descrição |
|---|---|
| `risk_probability` | Probabilidade calibrada (escala real: 0.28–0.89) |
| `risk_label` | Classificação binária pelo threshold 0.40 |
| `clinical_risk_level` | Nível clínico em 5 faixas relativas à escala do modelo |
| `margin_to_threshold` | Distância ao corte (positivo = acima, negativo = abaixo) |
| `top_risk_factors` | Top 5 fatores que aumentaram o risco (SHAP > 0) |
| `top_protective_factors` | Top 5 fatores que reduziram o risco (SHAP < 0) |
| `interpretation_warnings` | Contexto de escala: teto (~0.89), zona cinza (0.35–0.40), recall |

**Escala de risco clínico:**

| `clinical_risk_level` | Faixa de probabilidade | Referência |
|---|---|---|
| `muito_alto` | ≥ 0.70 | Top 10% dos prematuros reais |
| `alto` | 0.50 – 0.70 | Acima da mediana dos prematuros reais |
| `moderado` | 0.40 – 0.50 | Acima do threshold, margem estreita |
| `zona_cinza` | 0.35 – 0.40 | Abaixo do threshold — ~12% dos FN caem aqui |
| `baixo` | < 0.35 | Abaixo do p5 dos prematuros reais |

> **Nota sobre o teto:** o modelo tem probabilidade máxima empírica de ~0.89 (limitado pelo sinal disponível no SINASC/MG). Uma predição de 0.70 já representa 61% da margem disponível entre threshold e teto — contexto que `interpretation_warnings` fornece automaticamente à LLM.

## Insumos importados da Fase 1

* `data/X_train.parquet`, `data/X_test.parquet`, `data/y_train.parquet`, `data/y_test.parquet`: splits estratificados já preparados.
* `results/artifacts/best_model_calibrated.pkl`: melhor modelo da Fase 1, calibrado, usado como baseline de comparação para o AG.
* `results/metrics/best_model_operational_metrics.json`: hiperparâmetros do baseline (ponto de partida do AG) e métricas operacionais.
* `results/metrics/*.csv` e `experiment_config.json`: métricas detalhadas e configuração do experimento Fase 1, usadas no comparativo.

Cenário fixado: B (29 features ordinais e contínuas), `sample_weight=balanced`, threshold operacional 0.40 com piso clínico de recall maior ou igual a 0.80.

## Experimentos AG conduzidos (Etapa 1, Isa)

Os 6 notebooks varrem configurações diferentes do AG sobre o problema de prematuridade. A síntese e os trade-offs estão em `results/AG_resultados_resumo.md`. Métricas detalhadas em `results/metrics/01_ag_comparison*.csv`, `02_ag_comparison_*.csv`, `03_comparison_*.csv`, `04_comparison_*.csv`, `05_comparison_*.csv` e `06_ag_coevo_comparison.csv`. Curvas de convergência em `results/figures/`.

| Notebook | Experimento | Foco |
|---|---|---|
| 01 | AG base | 6 genes, recall e F2 com piso clínico |
| 02 | Threshold 0.40 | mesmo AG fixando o threshold operacional |
| 03 | Expanded search | espaço de busca com 8 genes |
| 04 | SMOTE | balanceamento sintético da classe positiva |
| 05 | Random Forest | troca o estimador (HGB para RF) |
| 06 | Co-evolução | evolui hiperparâmetros e threshold em conjunto |

 ## Escalabilidade, monitoramento e logging

A estratégia de escalabilidade automática, monitoramento e logging está documentada em:

```text
docs/scalabilidade_monitoramento_arquitetura.md
```

O projeto também inclui um módulo utilitário para logs estruturados e tracking de desempenho:

```text
src/utils/monitoring.py
```

Esse módulo permite registrar:

* eventos operacionais;
* tempo de execução de treinos e inferências;
* métricas dos modelos;
* métricas por geração dos Algoritmos Genéticos;
* logs em formato JSON Lines.

Os arquivos gerados são salvos em:

```text
results/logs/application.jsonl
results/metrics/performance_metrics.jsonl
```

## Evidência de teste do monitoramento

Foi criado o script manual de validação:

```text
src/test_monitoring_manual.py
```

O teste pode ser executado com:

```bash
uv run python src/test_monitoring_manual.py
```

Resultado esperado no terminal:

```text
Teste manual de monitoramento executado com sucesso.
```

A execução valida:

* geração de eventos operacionais;
* registro de duração de processamento;
* registro de inferência simulada;
* registro de métricas de modelo;
* registro de métricas de geração do Algoritmo Genético.

Arquivos gerados automaticamente após a execução:

```text
results/logs/application.jsonl
results/metrics/performance_metrics.jsonl
```
```