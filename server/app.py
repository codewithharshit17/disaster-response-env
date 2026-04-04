from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import os

from models import Action
from server.env import DisasterResponseEnv

app = FastAPI()

env = DisasterResponseEnv()


# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok"}


# -----------------------------
# RESET (GET + POST)
# -----------------------------
@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    result = env.reset()
    return JSONResponse(content=result.dict())


# -----------------------------
# STEP (GET + POST)
# -----------------------------
@app.api_route("/step", methods=["GET", "POST"])
def step(action: Action):
    result = env.step(action)
    return JSONResponse(content=result.dict())


# -----------------------------
# STATE
# -----------------------------
@app.get("/state")
def state():
    return JSONResponse(content=env.state.dict())


# -----------------------------
# MAIN (for OpenEnv)
# -----------------------------
def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()