import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://codewithharshit17-disaster-response-env.hf.space"
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url="https://api.openai.com/v1", api_key=API_KEY)

TASK_NAME = "easy"
ENV_NAME = "disaster_response"
MAX_STEPS = 5


# -----------------------------
# LOGGING
# -----------------------------
def log_start():
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}"
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


# Safe extractor
def extract_observation(data):
    if isinstance(data, dict) and "observation" in data:
        return data["observation"]
    return data


def get_action(obs):
    regions = obs.get("regions", [])

    if not regions:
        return {"action_type": "allocate", "region_name": "A"}

    available = [r for r in regions if not r.get("helped", False)]

    if not available:
        return {
            "action_type": "allocate",
            "region_name": regions[0].get("name", "A")
        }

    best = max(available, key=lambda r: r.get("severity", 0))

    return {
        "action_type": "allocate",
        "region_name": best.get("name", "A")
    }


def main():
    rewards = []

    log_start()

    try:
        # RESET
        res = requests.post(f"{API_BASE_URL}/reset", json={})
        data = res.json()

        obs = extract_observation(data)

        for step in range(1, MAX_STEPS + 1):
            action = get_action(obs)

            res = requests.post(f"{API_BASE_URL}/step", json=action)
            data = res.json()

            #SAFE EXTRACTION
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))

            rewards.append(reward)

            log_step(step, action, reward, done, None)

            obs = extract_observation(data)

            if done:
                break

        success = sum(rewards) > 0

    except Exception as e:
        log_step(0, "error", 0.0, True, str(e))
        success = False

    log_end(success, len(rewards), rewards)


if __name__ == "__main__":
    main()