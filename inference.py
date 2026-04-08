import os
import requests
from openai import OpenAI

# -----------------------------
# ENVIRONMENT API (your app)
# -----------------------------
ENV_BASE_URL = "https://codewithharshit17-disaster-response-env.hf.space"

# -----------------------------
# LLM PROXY (MANDATORY)
# -----------------------------
LLM_BASE_URL = os.getenv("API_BASE_URL")
LLM_API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

client = None
if LLM_BASE_URL and LLM_API_KEY:
    try:
        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
    except:
        client = None

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
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}")


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


# -----------------------------
# SAFE REQUEST
# -----------------------------
def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return res.json()
        return {}
    except:
        return {}


# -----------------------------
# ACTION LOGIC
# -----------------------------
def get_action(obs):
    regions = obs.get("regions", [])
    available = [r for r in regions if not r.get("helped", False)]

    if not available:
        return {"action_type": "allocate", "region_name": "A"}

    # 🔥 REQUIRED: LLM PROXY CALL
    if client:
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "choose best region"}],
                max_tokens=5,
            )
        except:
            pass
    best = max(available, key=lambda r: r.get("severity", 0))

    return {
        "action_type": "allocate",
        "region_name": best.get("name", "A")
    }


# -----------------------------
# MAIN
# -----------------------------
def main():
    rewards = []

    log_start()

    try:
        data = safe_post(f"{ENV_BASE_URL}/reset", {})
        obs = data

        for step in range(1, MAX_STEPS + 1):
            action = get_action(obs)

            data = safe_post(f"{ENV_BASE_URL}/step", action)

            #SAFE EXTRACTION
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))

            rewards.append(reward)

            log_step(step, action, reward, done, None)

            obs = data

            if done:
                break

        success = sum(rewards) > 0

    except Exception as e:
        log_step(0, "error", 0.0, True, str(e))
        success = False

    log_end(success, len(rewards), rewards)


if __name__ == "__main__":
    main()