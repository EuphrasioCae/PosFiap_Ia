from __future__ import annotations

import json
from typing import Any

from src.schemas import RespostaPredicao


def build_llm_context(resp: RespostaPredicao) -> dict[str, Any]:
    """Convert a prediction response into a JSON-serializable LLM context.

    The output follows the contract in docs/pipeline_plan.md and is safe to
    pass directly to the prompt builder and LLM client.
    """
    data = resp.model_dump(mode="json")
    # Validate serializability early — fail fast before calling the LLM.
    json.dumps(data, ensure_ascii=False)
    return data
