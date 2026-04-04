from fastapi import FastAPI
from models import Observation, Action
from server import env

app = FastAPI()


@app.post("/reset", response_model=Observation)
def reset():
    return env.reset()


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = env.step(action)

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=Observation)
def state():
    return env._get_obs()