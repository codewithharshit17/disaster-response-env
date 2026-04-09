# tasks/tasks.py

REGIONS = [
    {"name": "A", "severity": 0.9, "population": 1000},
    {"name": "B", "severity": 0.5, "population": 500},
    {"name": "C", "severity": 0.7, "population": 800},
]
SEVERITY = {r["name"]: r["severity"] for r in REGIONS}
OPTIMAL_ORDER = ["A", "C", "B"]  # sorted by severity desc


def clamp(score: float) -> float:
    """
    Clamp score to strictly between (0, 1).
    - if score <= 0 ? return 0.1
    - if score >= 1 ? return 0.9
    - else return score
    """
    if score <= 0.0:
        return 0.1
    elif score >= 1.0:
        return 0.9
    else:
        return score


def grade_easy(history: list) -> float:
    """
    Easy: Did the agent pick the highest-severity region first?
    Partial credit based on severity of first pick.
    """
    if not history:
        return 0.1

    first_pick = history[0]
    severity_of_first = SEVERITY.get(first_pick, 0.0)
    max_severity = max(SEVERITY.values())  # 0.9

    raw = severity_of_first / max_severity  # e.g. 1.0 if picked A, 0.78 if C, 0.56 if B
    return clamp(raw)


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
    return clamp(raw)


def grade_hard(history: list) -> float:
    """
    Hard: Did the agent cover all regions in optimal order (A ? C ? B)?
    Penalize wasted moves (revisits) and wrong ordering.
    """
    if not history:
        return 0.1

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
    return clamp(raw)


def grade_advanced(history: list) -> float:
    """
    Advanced: Cover A and C, but avoid B (e.g., assuming B is inaccessible or too low priority).
    """
    if not history:
        return 0.1
    
    score = 0.0
    if "A" in history: score += 0.4
    if "C" in history: score += 0.4
    if "B" not in history: 
        score += 0.2  # Bonus for avoiding B
    else:
        score -= 0.3  # Penalty for visiting B
        
    return clamp(score)


def grade_expert(history: list) -> float:
    """
    Expert: Resolve the two highest severity regions (A and C) in exactly 2 steps.
    """
    if not history:
        return 0.1
        
    # Perfect run
    if len(history) == 2 and set(history) == {"A", "C"}:
        return 0.9
        
    # Covered both but took too many steps
    if "A" in history and "C" in history:
        return 0.5
        
    # Covered only one
    if "A" in history or "C" in history:
        return 0.3
        
    return clamp(0.1)


TASKS = {
    "easy": {
        "description": "Allocate ambulance to highest severity region",
    },
    "medium": {
        "description": "Handle multiple regions efficiently with full coverage",
    },
    "hard": {
        "description": "Optimize full disaster response in correct priority order",
    },
    "advanced": {
        "description": "Cover severe regions while avoiding low-priority region B",
    },
    "expert": {
        "description": "Resolve the highest severity regions in exactly 2 steps",
    },
}