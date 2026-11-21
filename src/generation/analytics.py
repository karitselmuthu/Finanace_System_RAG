from __future__ import annotations

from collections import defaultdict


def analyze(
    rows: list[dict],
    txn_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    keyword: str | None = None,
    source_doc: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    group_by: str | None = None,
    sort: str = "desc",
    limit: int = 10,
) -> dict:
    """Filter extracted transactions and compute stats. Dates are ISO strings."""

    def ok(t: dict) -> bool:
        a, d = float(t["amount"]), t["transaction_date"]
        return (
            (not txn_type or t["transaction_type"] == txn_type)
            and (not start_date or d >= start_date)
            and (not end_date or d <= end_date)
            and (not keyword or keyword.lower() in t["description"].lower())
            and (not source_doc or source_doc.lower() in (t.get("source_doc") or "").lower())
            and (min_amount is None or a >= min_amount)
            and (max_amount is None or a <= max_amount)
        )

    m = [t for t in rows if ok(t)]
    amts = [float(t["amount"]) for t in m]
    out = {
        "matched": len(m),
        "sum": round(sum(amts), 2),
        "avg": round(sum(amts) / len(amts), 2) if amts else 0,
        "max": max(amts, default=0),
        "min": min(amts, default=0),
    }
    if group_by:  # month | year | type | document
        n = {"month": 7, "year": 4}.get(group_by)
        g: dict[str, list] = defaultdict(lambda: [0.0, 0])
        for t in m:
            if group_by == "document":
                k = t.get("source_doc") or "unknown"
            else:
                k = t["transaction_date"][:n] if n else t["transaction_type"]
            g[k][0] += float(t["amount"])
            g[k][1] += 1
        out["groups"] = {k: {"sum": round(v[0], 2), "count": v[1]} for k, v in sorted(g.items())}
    m.sort(key=lambda t: float(t["amount"]), reverse=sort != "asc")
    out["rows"] = m[: max(1, min(limit, 50))]
    return out


if __name__ == "__main__":
    R = [
        {"transaction_date": "2024-01-05", "description": "Amazon", "amount": "100", "transaction_type": "debit"},
        {"transaction_date": "2024-01-20", "description": "Salary", "amount": "1000", "transaction_type": "credit"},
        {"transaction_date": "2024-02-02", "description": "AMAZON PAY", "amount": "50.5", "transaction_type": "debit"},
    ]
    r = analyze(R, txn_type="debit", keyword="amazon", group_by="month")
    assert r["sum"] == 150.5 and r["matched"] == 2 and r["groups"]["2024-02"]["sum"] == 50.5
    assert analyze(R, start_date="2024-02-01")["matched"] == 1
    assert analyze(R, limit=1)["rows"][0]["description"] == "Salary"
    R[0]["source_doc"], R[1]["source_doc"], R[2]["source_doc"] = "a.pdf", "b.pdf", "a.pdf"
    assert analyze(R, source_doc="A.pdf")["matched"] == 2
    assert analyze(R, group_by="document")["groups"]["b.pdf"]["sum"] == 1000
    print("ok")
