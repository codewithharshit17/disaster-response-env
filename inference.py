import os
import time
import requests
from openai import OpenAI

# -----------------------------
# ENVIRONMENT API
# The validator injects ENV_BASE_URL pointing to your running container.
# Fall back to the HF Space URL only if not set.
# -----------------------------
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://codewithharshit17-disaster-response-env.hf.space").rstrip("/")

# -----------------------------
# LLM PROXY (MANDATORY)
# The validator injects these. Do NOT hardcode keys.
# -----------------------------
LLM_BASE_URL = os.environ["API_BASE_URL"]   # raises KeyError if missing — intentional
LLM_API_KEY  = os.environ["API_KEY"]        # raises KeyError if missing — intentional
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Initialize OpenAI client pointed at the hackathon proxy
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

TASK_NAME = os.getenv("TASK_NAME", "easy")
ENV_NAME  = "disaster_response"
MAX_STEPS = 10


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
# WAIT FOR ENV TO BE READY
# -----------------------------
def wait_for_env(timeout=60):
    print(f"[INFO] Waiting for env at {ENV_BASE_URL} ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print("[INFO] Env is ready.")
                return True
        except Exception:
            pass
        time.sleep(3)
    print("[WARN] Env did not become ready in time, proceeding anyway.")
    return False


# -----------------------------
# SAFE HTTP REQUEST
# -----------------------------
def safe_post(url, payload, retries=3):
    for attempt in range(retries):
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                return res.json()
            print(f"[WARN] POST {url} returned {res.status_code}")
        except Exception as e:
            print(f"[WARN] POST {url} attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return {}


# -----------------------------
# LLM-POWERED ACTION SELECTION
# This is the mandatory proxy call — do NOT wrap in bare except/pass.
# The LLM response is parsed and used to pick the region.
# -----------------------------
def get_action_from_llm(obs: dict) -> dict:
    regions = obs.get("regions", [])
    available = [r for r in regions if not r.get("helped", False)]

    if not available:
        return {"action_type": "allocate", "region_name": regions[0]["name"] if regions else "A"}

    # Build a prompt describing the situation
    region_desc = "\n".join(
        f"- Region {r['name']}: severity={r['severity']:.2f}, population={r['population']}, helped={r['helped']}"
        for r in available
    )
    prompt = (
        f"You are coordinating disaster response. Available regions:\n{region_desc}\n\n"
        f"Which single region should receive an ambulance next? "
        f"Reply with ONLY the region name (one letter, e.g. A or B or C). No explanation."
    )

    # MANDATORY: call through the hackathon LLM proxy — no try/except swallowing
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )

    chosen = response.choices[0].message.content.strip().upper()

    # Validate the LLM's answer — fall back to highest severity if bad output
    valid_names = {r["name"] for r in available}
    if chosen not in valid_names:
        best = max(available, key=lambda r: r.get("severity", 0))
        chosen = best["name"]

    return {"action_type": "allocate", "region_name": chosen}


# -----------------------------
# MAIN
# -----------------------------
def main():
    rewards = []
    log_start()

    wait_for_env()

    # Reset environment
    obs = safe_post(f"{ENV_BASE_URL}/reset", {})
    if not obs:
        print("[ERROR] Could not reset environment.")
        log_end(False, 0, [])
        return

    for step in range(1, MAX_STEPS + 1):
        try:
            action = get_action_from_llm(obs)
        except Exception as e:
            # LLM call failed — log it, fall back to heuristic, but don't silently pass
            print(f"[ERROR] LLM call failed at step {step}: {e}")
            regions = obs.get("regions", [])
            available = [r for r in regions if not r.get("helped", False)]
            if available:
                best = max(available, key=lambda r: r.get("severity", 0))
                action = {"action_type": "allocate", "region_name": best["name"]}
            else:
                action = {"action_type": "allocate", "region_name": "A"}

        data = safe_post(f"{ENV_BASE_URL}/step", action)

        reward = float(data.get("reward", 0.0))
        done   = bool(data.get("done", False))
        rewards.append(reward)

        log_step(step, action, reward, done, None)

        obs = data
        if done:
            break

    success = sum(rewards) > 0
    log_end(success, len(rewards), rewards)


if __name__ == "__main__":
    main()