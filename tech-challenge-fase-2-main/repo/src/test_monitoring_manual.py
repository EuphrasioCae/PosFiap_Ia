import time

from utils.monitoring import (
    log_event,
    log_metric,
    log_model_metrics,
    log_inference_result,
    log_genetic_algorithm_generation,
    track_duration,
)


def main() -> None:
    log_event(
        event="teste_monitoramento_manual",
        status="ok",
        componente="validacao",
    )

    log_metric(
        metric_name="teste_recall",
        metric_value=0.85,
        step=1,
        context={"experiment_name": "teste_monitoramento"},
    )

    with track_duration(
        "teste_bloco_processamento",
        componente="validacao",
    ):
        time.sleep(1)

    log_model_metrics(
        experiment_name="teste_modelo",
        metrics={
            "accuracy": 0.82,
            "recall": 0.86,
            "precision": 0.79,
            "f1_score": 0.80,
        },
        step=1,
    )

    log_inference_result(
        model_name="HistGradientBoostingClassifier",
        threshold=0.40,
        probability=0.73,
        prediction=1,
        duration_seconds=0.081,
        records_processed=1,
    )

    log_genetic_algorithm_generation(
        experiment_name="AG_teste_monitoramento",
        generation=1,
        best_fitness=0.8421,
        mean_fitness=0.8014,
        best_params={
            "learning_rate": 0.05,
            "max_iter": 200,
        },
    )

    print("Teste manual de monitoramento executado com sucesso.")


if __name__ == "__main__":
    main()