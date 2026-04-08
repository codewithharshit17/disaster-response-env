"""
server/app.py - FastAPI server for the Disaster Response Environment.
Uses OpenEnv framework's create_fastapi_app for proper grader wiring.
"""

import os
import sys

# Fix imports for both local and Docker environments
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server"))

from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app

# Import models and environment
from models import Action, Observation
from server.env import DisasterResponseEnv

# Create app using OpenEnv framework (automatically wires graders)
app = create_fastapi_app(DisasterResponseEnv, Action, Observation)


@app.get("/")
def root():
    return JSONResponse(content={
        "name": "Disaster Response Environment",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "grader": "/grader",
            "baseline": "/baseline",
            "tasks": "/tasks",
        }
    })


@app.get("/tasks", tags=["Competition"])
def get_tasks():
    """List all available tasks"""
    return JSONResponse(content={
        "tasks": [
            {
                "id": "easy",
                "description": "Allocate ambulance to highest severity region",
                "difficulty": "easy",
            },
            {
                "id": "medium",
                "description": "Handle multiple regions efficiently with full coverage",
                "difficulty": "medium",
            },
            {
                "id": "hard",
                "description": "Optimize full disaster response in correct priority order",
                "difficulty": "hard",
            },
        ],
        "action_schema": {
            "region_name": "string - region to allocate ambulance to (A, B, or C)",
        },
    })


def main():
    """Entry point for uv run server and project.scripts."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 1))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers, reload=False)


if __name__ == "__main__":
    main()
