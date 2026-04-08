from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os

from models import Action
from server.env import DisasterResponseEnv

try:
    from tasks.tasks import grade_easy, grade_medium, grade_hard
except ImportError:
    from graders.easy import grade_easy, grade_medium, grade_hard

app = FastAPI()
env: DisasterResponseEnv = None

REGIONS = [
    {"name": "A", "severity": 0.9, "population": 1000},
    {"name": "B", "severity": 0.5, "population": 500},
    {"name": "C", "severity": 0.7, "population": 800},
]


@app.on_event("startup")
def startup():
    global env
    env = DisasterResponseEnv()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok"}


@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    result = env.reset()
    return JSONResponse(content=result.dict())


@app.api_route("/step", methods=["GET", "POST"])
def step(action: Action):
    result = env.step(action)
    return JSONResponse(content=result.dict())


@app.get("/state")
def state():
    return JSONResponse(content=env.state.dict())


# --------------------------------------------------
# /grader — score a trajectory without a full episode
# The validator may POST a trajectory here directly.
# --------------------------------------------------
@app.post("/grader")
def grader(payload: dict):
    """
    Accepts: {"task_id": "easy"|"medium"|"hard", "history": ["A","C","B"]}
    OR:      {"task_id": ..., "trajectory": [...step dicts...]}
    Returns: {"score": float, "passed": bool, "feedback": str, "task_id": str}
    """
    task_id = payload.get("task_id", "easy").lower()

    # Support history as plain list or extracted from trajectory
    history = payload.get("history", [])
    if not history:
        trajectory = payload.get("trajectory", [])
        for step_data in trajectory:
            action = step_data.get("action", {})
            region = action.get("region_name") or action.get("region", "")
            if region:
                history.append(region)

    result = env.run_grader(task_id, history)
    return JSONResponse(content=result)


# --------------------------------------------------
# /baseline — run oracle agent on all 3 tasks and return scores
# The validator calls this to confirm graders work and scores are in (0,1).
# --------------------------------------------------
@app.get("/baseline")
def baseline():
    """
    Runs a perfect oracle agent on each task and returns scores.
    Oracle always picks in optimal order: A (0.9) → C (0.7) → B (0.5)
    """
    optimal_history = ["A", "C", "B"]

    results = {}
    scores = []
    
    for task_id in ["easy", "medium", "hard"]:
        result = env.run_grader(task_id, optimal_history)
        results[task_id] = result
        scores.append(result["score"])
        
        # Validate score is strictly in (0, 1)
        s = result["score"]
        assert 0 < s < 1, f"Score out of range for {task_id}: {s}"

    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return JSONResponse(content={
        "baseline_agent": "oracle (optimal order: A → C → B)",
        "results": results,
        "average_score": round(avg_score, 4)
    })


# --------------------------------------------------
# /tasks — list all tasks and their action schema
# --------------------------------------------------
@app.get("/tasks")
def tasks():
    return JSONResponse(content={
        "tasks": [
            {
                "id": "easy",
                "description": "Allocate ambulance to highest severity region",
                "difficulty": "easy",
                "scoring": "Partial credit based on severity of first pick",
            },
            {
                "id": "medium",
                "description": "Handle multiple regions efficiently with full coverage",
                "difficulty": "medium",
                "scoring": "Partial credit based on severity-weighted coverage",
            },
            {
                "id": "hard",
                "description": "Optimize full disaster response in correct priority order",
                "difficulty": "hard",
                "scoring": "Partial credit based on ordering and efficiency",
            },
        ],
        "action_schema": {
            "action_type": "string — 'allocate'",
            "region_name": "string — A, B, or C",
        },
    })


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()