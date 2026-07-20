from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from src.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 1

MOCK_RESPONSE = """\
**Resumo clinico**

Esta gestante apresenta probabilidade estimada de parto prematuro conforme o contexto
fornecido pelo modelo. A classificacao operacional e o nivel de risco clinico seguem
exatamente os valores do JSON de entrada.

**Fatores associados ao aumento do risco**

- Interpretacao mockada — configure OPENAI_API_KEY ou desative LLM_MOCK para resposta real.

**Fatores associados a reducao do risco**

- Nenhum fator protetor relevante identificado neste mock.

**Alertas de interpretacao**

- SHAP mede associacao do modelo, nao causalidade clinica.
- A resposta nao substitui avaliacao profissional.
"""


class LLMError(Exception):
    """Raised when the LLM call fails after retries."""


def _is_mock_mode() -> bool:
    return os.getenv("LLM_MOCK", "0").strip() in {"1", "true", "True", "yes"}


def generate_interpretation(context: dict[str, Any]) -> str:
    """Generate a clinical interpretation from structured model context.

    Uses OpenAI GPT-4o-mini by default. Set LLM_MOCK=1 to return a fixed response
    without calling the API (useful for tests and offline demos).
    """
    if _is_mock_mode():
        return MOCK_RESPONSE

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENAI_API_KEY nao configurada. Defina a variavel de ambiente ou use LLM_MOCK=1."
        )

    client = OpenAI(api_key=api_key, timeout=DEFAULT_TIMEOUT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context)},
    ]

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
                messages=messages,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMError("Resposta vazia da LLM.")
            return content.strip()
        except LLMError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.0)
                continue
            break

    raise LLMError(f"Falha ao chamar a LLM: {last_error}") from last_error
