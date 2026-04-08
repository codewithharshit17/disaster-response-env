def grade_easy(history):
    if not history:
        return 0.1  # was 0.0 ❌

    if history[0] == "A":
        return 0.9  # was 1.0 ❌

    return 0.2  # instead of 0.0


def grade_medium(history):
    if len(history) < 2:
        return 0.2  # was 0.0 ❌

    if set(history[:2]) == {"A", "C"}:
        return 0.85  # was 1.0 ❌

    return 0.5  # already valid ✅


def grade_hard(history):
    correct_order = ["A", "C", "B"]

    score = 0
    for i in range(min(len(history), 3)):
        if history[i] == correct_order[i]:
            score += 1

    raw_score = score / 3.0

    # 🔥 force into (0,1)
    if raw_score <= 0:
        return 0.3
    elif raw_score >= 1:
        return 0.95
    else:
        return raw_score