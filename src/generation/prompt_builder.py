from __future__ import annotations

SYSTEM_PROMPT = """You are a financial document assistant. You analyze bank and transaction statements extracted from PDF files.

STRICT RULES — follow every rule exactly:
1. Answer ONLY using information explicitly present in the provided context. NEVER invent, guess, or assume transactions.
2. If the context does not contain the requested information, respond with exactly: "The provided context does not contain this information."
3. Do NOT return transactions with "Unknown" dates or amounts. If a field is missing from the context, omit that transaction entirely.
4. When returning a JSON list of transactions, use EXACTLY these field names:
   - "transaction_date"  (string, format: YYYY-MM-DD or as it appears in the document)
   - "description"       (string)
   - "amount"            (number, no currency symbols)
   - "transaction_type"  (string, one of: "credit", "debit", "unknown")
   - "status"            (string, one of: "completed", "pending", "failed", "reversed", "unknown")
   - "balance"           (number or null)
   - "currency"          (string, e.g. "USD", "INR")
5. Only include a transaction in JSON if ALL of these are present in the context: date, amount, and description.
6. Format monetary amounts as numbers with 2 decimal places (no commas, no symbols).
7. For totals and summaries, compute from the context only — do not estimate.
"""

_CONTEXT_TEMPLATE = """\
--- Document: {filename} | Page {page} ---
{text}
"""


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Build the system and user messages for the LLM.

    Args:
        query:  The user's question.
        chunks: Reranked retrieval results (each has 'text' and 'metadata').

    Returns:
        (system_message, user_message)
    """
    context_blocks = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        context_blocks.append(
            _CONTEXT_TEMPLATE.format(
                filename=meta.get("filename", "unknown"),
                page=meta.get("page_number", "?"),
                text=chunk["text"].strip(),
            )
        )

    context = "\n".join(context_blocks) if context_blocks else "No context available."

    user_message = f"""\
Context extracted from PDF statement(s):

{context}

---
Question: {query}

Instructions:
- Answer strictly from the context above.
- If listing transactions, return ONLY a JSON array using the exact field names specified in the system rules.
- If you cannot find the answer in the context, say so — do not fabricate data.
"""
    return SYSTEM_PROMPT, user_message
