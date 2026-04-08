from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os

from models import Action
from server.env import DisasterResponseEnv

app = FastAPI()
env: DisasterResponseEnv = None


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


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()