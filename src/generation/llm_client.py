from __future__ import annotations

import json
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.prompt_builder import build_prompt
from src.utils.config import settings
from src.utils.logger import logger

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(system: str, user: str) -> str:
    response = _get_client().chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _extract_json(text: str) -> list | dict | None:
    """Try to parse a JSON array or object from the LLM response."""
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    for start, end in [("[", "]"), ("{", "}")]:
        idx = text.find(start)
        if idx != -1:
            try:
                return json.loads(text[idx: text.rfind(end) + 1])
            except json.JSONDecodeError:
                pass
    return None


_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze",
        "description": (
            "Filter the user's extracted transactions and compute sum/avg/min/max/count, "
            "optional grouping, and the top matching rows. All filters optional."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "txn_type": {"type": "string", "enum": ["credit", "debit"]},
                "start_date": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD inclusive"},
                "keyword": {"type": "string", "description": "case-insensitive match in description"},
                "min_amount": {"type": "number"},
                "max_amount": {"type": "number"},
                "source_doc": {"type": "string", "description": "restrict to one PDF (filename or part of it)"},
                "group_by": {"type": "string", "enum": ["month", "year", "type", "document"]},
                "sort": {"type": "string", "enum": ["desc", "asc"], "description": "by amount"},
                "limit": {"type": "integer", "description": "rows to return, max 50"},
            },
        },
    },
}

_TOOL_SYSTEM = (
    "You answer questions about bank/UPI transactions. NEVER do arithmetic yourself: call the "
    "analyze tool (as many times as needed) and report its numbers exactly. Sum/avg across "
    "credit+debit mixed is meaningless, so set txn_type or group_by='type'. Dates in the data "
    "span many years; filter by date when the user names a period. Do not assume a currency "
    "symbol. Be concise. Data comes from separate statements (documents) that may be different "
    "accounts: if the user doesn't name one and several exist, don't blend them silently - "
    "use group_by='document' (or one call per document) and report each, saying which is which."
)


def answer_with_tools(query: str, rows: list[dict]) -> dict:
    """Answer using exact calculations over extracted transactions (function calling)."""
    from src.generation.analytics import analyze

    docs = {}
    for r in rows:
        docs[r.get("source_doc") or "unknown"] = docs.get(r.get("source_doc") or "unknown", 0) + 1
    messages = [
        {"role": "system", "content": _TOOL_SYSTEM + f" Documents (name: transaction count): {docs}"},
        {"role": "user", "content": query},
    ]
    last_rows: list[dict] = []
    for _ in range(5):
        msg = _get_client().chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=messages,
            tools=[_TOOL],
        ).choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "transactions": last_rows}
        messages.append(msg)
        for call in msg.tool_calls:
            try:
                result = analyze(rows, **json.loads(call.function.arguments))
                last_rows = result["rows"]
            except Exception as e:  # bad args from the model: let it retry
                result = {"error": f"{type(e).__name__}: {e}"}
            logger.info(f"tool analyze({call.function.arguments}) -> {str(result)[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)}
            )
    return {"answer": "Could not complete the calculation.", "transactions": last_rows}


def answer_query(query: str, chunks: list[dict]) -> dict:
    """
    Send a query + retrieved chunks to the LLM and return a structured response.

    Returns:
        {
            "answer":       str  — plain-language answer,
            "transactions": list — parsed JSON transactions if present,
            "raw":          str  — raw LLM text output,
        }
    """
    system, user = build_prompt(query, chunks)
    logger.info(f"LLM query: '{query[:80]}'")

    raw = _call_llm(system, user)
    logger.debug(f"LLM raw response length: {len(raw)} chars")

    transactions = _extract_json(raw) or []
    if isinstance(transactions, dict):
        transactions = [transactions]

    return {
        "answer": raw,
        "transactions": transactions,
        "raw": raw,
    }
