# graders/easy.py

REGIONS = [
    {"name": "A", "severity": 0.9, "population": 1000},
    {"name": "B", "severity": 0.5, "population": 500},
    {"name": "C", "severity": 0.7, "population": 800},
]
SEVERITY = {r["name"]: r["severity"] for r in REGIONS}
OPTIMAL_ORDER = ["A", "C", "B"]  # sorted by severity desc


def grade_easy(history: list) -> float:
    """
    Easy: Did the agent pick the highest-severity region first?
    Partial credit based on severity of first pick.
    """
    if not history:
        return 0.05

    first_pick = history[0]
    severity_of_first = SEVERITY.get(first_pick, 0.0)
    max_severity = max(SEVERITY.values())  # 0.9

    raw = severity_of_first / max_severity  # e.g. 1.0 if picked A, 0.78 if C, 0.56 if B
    return max(0.01, min(0.99, raw * 0.98))  # ensure never exactly 1.0


def grade_medium(history: list) -> float:
    """
    Medium: Did the agent cover all regions, weighted by severity?
    """
    if not history:
        return 0.1

    helped = {}
    for name in history:
        if name not in helped and name in SEVERITY:
            helped[name] = SEVERITY[name]

    total_severity = sum(SEVERITY.values())
    covered_severity = sum(helped.values())

    raw = covered_severity / total_severity
    return max(0.01, min(0.99, raw))


def grade_hard(history: list) -> float:
    """
    Hard: Did the agent cover all regions in optimal order (A → C → B)?
    Penalize wasted moves (revisits) and wrong ordering.
    """
    if not history:
        return 0.05

    # Order score: compare agent's unique order to optimal
    seen = []
    for name in history:
        if name not in seen and name in SEVERITY:
            seen.append(name)

    order_score = 0.0
    for i, name in enumerate(seen):
        if i < len(OPTIMAL_ORDER) and name == OPTIMAL_ORDER[i]:
            order_score += 1.0
        else:
            order_score += 0.3  # partial credit for visiting but wrong order

    max_order_score = len(OPTIMAL_ORDER) * 1.0

    # Waste penalty: revisits
    revisits = len(history) - len(seen)
    waste_penalty = revisits * 0.1

    raw = (order_score / max_order_score) - waste_penalty
    return max(0.01, min(0.99, raw))