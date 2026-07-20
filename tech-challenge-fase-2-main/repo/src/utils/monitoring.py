from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# Utilitários de monitoramento e logging estruturado para o Tech Challenge Fase 2.

# Este módulo fornece funções simples para registrar eventos operacionais,
# métricas de desempenho e duração de execuções em arquivos JSON Lines.

# A proposta é manter uma solução leve, sem dependências externas, adequada para:
# - notebooks;
# - scripts Python;
# - futura API de inferência;
# - jobs de otimização com Algoritmos Genéticos.

# Arquivos gerados:
# - results/logs/application.jsonl
# - results/metrics/performance_metrics.jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "results" / "logs"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

APPLICATION_LOG_FILE = LOG_DIR / "application.jsonl"
PERFORMANCE_METRICS_FILE = METRICS_DIR / "performance_metrics.jsonl"


def _utc_now_iso() -> str:
    """
    Retorna o timestamp atual em UTC no formato ISO 8601.
    """
    return datetime.now(timezone.utc).isoformat()


def _ensure_output_dirs() -> None:
    """
    Garante que os diretórios de logs e métricas existam.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> str:
    """
    Serializador auxiliar para objetos não suportados nativamente pelo JSON.
    """
    return str(value)


def append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    """
    Adiciona um registro em formato JSON Lines ao arquivo informado.
    """
    _ensure_output_dirs()

    with file_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=_json_default,
            )
            + "\n"
        )


def log_event(
    event: str,
    level: str = "INFO",
    **kwargs: Any,
) -> None:
    """
    Registra um evento operacional em JSON Lines.

    Exemplo:
        log_event(
            event="model_inference_finished",
            duration_seconds=0.083,
            records_processed=1,
            model_name="HistGradientBoostingClassifier",
        )
    """
    payload = {
        "timestamp": _utc_now_iso(),
        "level": level.upper(),
        "event": event,
        **kwargs,
    }

    append_jsonl(APPLICATION_LOG_FILE, payload)


def log_metric(
    metric_name: str,
    metric_value: float | int | str | bool | None,
    step: int | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Registra uma métrica de desempenho ou de modelo.

    Exemplo:
        log_metric(
            metric_name="recall",
            metric_value=0.84,
            step=10,
            context={"experiment": "AG_base"},
        )
    """
    payload = {
        "timestamp": _utc_now_iso(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "step": step,
        "context": context or {},
    }

    append_jsonl(PERFORMANCE_METRICS_FILE, payload)


@contextmanager
def track_duration(
    event: str,
    level: str = "INFO",
    **kwargs: Any,
) -> Generator[None, None, None]:
    """
    Context manager para medir duração de blocos de código.

    Exemplo:
        with track_duration("model_training", model_name="HGB"):
            model.fit(X_train, y_train)
    """
    start_time = time.perf_counter()

    log_event(
        event=f"{event}_started",
        level=level,
        **kwargs,
    )

    try:
        yield
    except Exception as exc:
        duration_seconds = time.perf_counter() - start_time

        log_event(
            event=f"{event}_failed",
            level="ERROR",
            duration_seconds=round(duration_seconds, 6),
            error_type=type(exc).__name__,
            error_message=str(exc),
            **kwargs,
        )

        raise
    else:
        duration_seconds = time.perf_counter() - start_time

        log_event(
            event=f"{event}_finished",
            level=level,
            duration_seconds=round(duration_seconds, 6),
            **kwargs,
        )


def configure_python_logging(level: int = logging.INFO) -> None:
    """
    Configura o logging padrão do Python para exibir mensagens no console.

    Esta função é opcional. Os logs estruturados principais são registrados
    pelos métodos `log_event`, `log_metric` e `track_duration`.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_model_metrics(
    experiment_name: str,
    metrics: dict[str, float | int | str | bool | None],
    step: int | None = None,
) -> None:
    """
    Registra um conjunto de métricas de modelo de uma vez.

    Exemplo:
        log_model_metrics(
            experiment_name="AG_base",
            metrics={
                "accuracy": 0.81,
                "recall": 0.84,
                "f1_score": 0.79,
            },
            step=5,
        )
    """
    for metric_name, metric_value in metrics.items():
        log_metric(
            metric_name=metric_name,
            metric_value=metric_value,
            step=step,
            context={"experiment_name": experiment_name},
        )


def log_inference_result(
    model_name: str,
    threshold: float,
    probability: float,
    prediction: int,
    duration_seconds: float,
    records_processed: int = 1,
) -> None:
    """
    Registra o resultado operacional de uma inferência.

    Observação:
    Este método não deve registrar dados sensíveis de pacientes.
    Apenas informações técnicas e estatísticas devem ser salvas.
    """
    log_event(
        event="model_inference_finished",
        model_name=model_name,
        threshold=threshold,
        probability=round(float(probability), 6),
        prediction=int(prediction),
        duration_seconds=round(float(duration_seconds), 6),
        records_processed=records_processed,
    )


def log_genetic_algorithm_generation(
    experiment_name: str,
    generation: int,
    best_fitness: float,
    mean_fitness: float | None = None,
    best_params: dict[str, Any] | None = None,
) -> None:
    """
    Registra informações de uma geração do Algoritmo Genético.
    """
    log_event(
        event="genetic_algorithm_generation_finished",
        experiment_name=experiment_name,
        generation=generation,
        best_fitness=round(float(best_fitness), 6),
        mean_fitness=round(float(mean_fitness), 6) if mean_fitness is not None else None,
        best_params=best_params or {},
    )

    log_metric(
        metric_name="best_fitness",
        metric_value=round(float(best_fitness), 6),
        step=generation,
        context={"experiment_name": experiment_name},
    )

    if mean_fitness is not None:
        log_metric(
            metric_name="mean_fitness",
            metric_value=round(float(mean_fitness), 6),
            step=generation,
            context={"experiment_name": experiment_name},
        )