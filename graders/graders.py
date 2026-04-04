def grade_easy(history):
    if not history:
        return 0.0

    if history[0] == "A":
        return 1.0
    return 0.0


def grade_medium(history):
    if len(history) < 2:
        return 0.0

    if set(history[:2]) == {"A", "C"}:
        return 1.0

    return 0.5


def grade_hard(history):
    correct_order = ["A", "C", "B"]

    score = 0
    for i in range(min(len(history), 3)):
        if history[i] == correct_order[i]:
            score += 1

    return score / 3.0