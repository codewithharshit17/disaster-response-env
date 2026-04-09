import os
from typing import List
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import Observation, Action, Region
except ImportError:
    from ..models import Observation, Action, Region

try:
    from tasks.tasks import grade_easy, grade_medium, grade_hard
except ImportError:
    from ..tasks.tasks import grade_easy, grade_medium, grade_hard


class DisasterResponseEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.regions: List[Region] = []
        self.ambulances: int = 0
        self.done: bool = False
        self.history: List[str] = []
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ===== Task & Grader Discovery =====
    @staticmethod
    def list_tasks():
        """Return list of all task definitions"""
        return [
            {"id": "easy", "description": "Allocate ambulance to highest severity region"},
            {"id": "medium", "description": "Handle multiple regions with full coverage"},
            {"id": "hard", "description": "Optimize disaster response with correct priority"},
        ]

    # ===== Episode Control =====
    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self) -> Observation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.regions = [
            Region(name="A", severity=0.9, population=1000, helped=False),
            Region(name="B", severity=0.5, population=500, helped=False),
            Region(name="C", severity=0.7, population=800, helped=False),
        ]

        self.time_remaining = 8
        self.step_count = 0
        self.ambulances = 3
        self.done = False
        self.history = []

        # Use actual graded score instead of hardcoded value
        task_name = os.getenv("TASK_NAME", "easy").lower()
        if "hard" in task_name:
            initial_score = grade_hard(self.history)
        elif "medium" in task_name:
            initial_score = grade_medium(self.history)
        else:
            initial_score = grade_easy(self.history)

        return Observation(
            regions=self.regions,
            ambulances=self.ambulances,
            score=initial_score,
            metadata={"score": initial_score}
        )

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: Action) -> Observation:
        self.step_count += 1
        self.time_remaining -= 1
        self._state.step_count += 1
        reward = 0.0

        if self.done:
            return self._get_obs(reward=0.0, done=True, score=0.05)

        # Find region
        target = None
        for r in self.regions:
            if r.name == action.region_name:
                target = r
                break

        # Invalid action
        if target is None or self.ambulances <= 0:
            reward = -1.0
        else:
            self.history.append(target.name)

            if target.helped:
                reward = -0.5
            else:
                severity_score = target.severity
                population_score = target.population / 1000
                reward = 0.7 * severity_score + 0.3 * population_score
                target.helped = True
                self.ambulances -= 1

        # Time penalty
        reward -= 0.05

        # Done condition
        if self.time_remaining <= 0 or self.ambulances == 0 or all(r.helped for r in self.regions):
            self.done = True

        # Bonus for finishing all regions
        if all(r.helped for r in self.regions):
            reward += 1.0

        # Grading — score goes into top-level field AND metadata
        task_name = os.getenv("TASK_NAME", "easy").lower()
        if "hard" in task_name:
            score = grade_hard(self.history)
        elif "medium" in task_name:
            score = grade_medium(self.history)
        else:
            score = grade_easy(self.history)

        return self._get_obs(reward=reward, done=self.done, score=score)

    # -----------------------------
    # GRADING
    # -----------------------------
    def run_grader(self, task_id: str, history: list) -> dict:
        """
        Grade a trajectory (history of region actions) for a specific task.
        Returns: {"score": float, "passed": bool, "feedback": str, "task_id": str}
        """
        task_id = task_id.lower()
        
        if task_id == "hard":
            score = grade_hard(history)
        elif task_id == "medium":
            score = grade_medium(history)
        else:
            score = grade_easy(history)
        
        passed = score > 0.5
        
        # Generate feedback based on task and score
        if task_id == "easy":
            feedback = f"Easy task: picked region {history[0] if history else 'none'} first (score: {score:.2f})"
        elif task_id == "medium":
            feedback = f"Medium task: covered {len(set(history) & {'A', 'B', 'C'})} regions (score: {score:.2f})"
        else:
            feedback = f"Hard task: ordering and efficiency score {score:.2f}"
        
        return {
            "score": score,
            "passed": passed,
            "feedback": feedback,
            "task_id": task_id
        }

    # -----------------------------
    # OBSERVATION & STATE
    # -----------------------------
    def _get_obs(self, reward=0.0, done=False, score=0.05) -> Observation:
        return Observation(
            regions=self.regions,
            ambulances=self.ambulances,
            reward=reward,
            done=done,
            score=score,           # top-level: read by deterministic grader
            metadata={"score": score}  # keep for backwards compat
        )

    @property
    def state(self) -> State:
        return self._state