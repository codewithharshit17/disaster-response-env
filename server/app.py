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
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state (GET)",
            "grader": "/grader (POST)",
            "baseline": "/baseline (GET)",
            "tasks": "/tasks (GET)",
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


@app.post("/test_inference", tags=["Testing"])
def test_inference():
    """Test endpoint to verify LLM proxy integration (Phase 2 requirement)"""
    try:
        from openai import OpenAI
        
        # Read the injected proxy credentials
        api_base = os.environ.get("API_BASE_URL")
        api_key = os.environ.get("API_KEY")
        
        if not api_base or not api_key:
            return JSONResponse(status_code=400, content={
                "error": "API_BASE_URL or API_KEY not provided"
            })
        
        # Initialize client with provided credentials
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        
        # Make a test API call through the proxy
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
            messages=[
                {"role": "user", "content": "Which region A, B, or C should get priority? Reply with just the letter."}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "LLM proxy integration verified",
            "llm_response": response.choices[0].message.content,
            "api_base_used": api_base,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": str(e),
            "message": "Failed to call LLM through proxy"
        })


@app.get("/verify_llm", tags=["Testing"])
def verify_llm():
    """Verify LLM proxy is configured and accessible"""
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("API_KEY")
    
    return JSONResponse(content={
        "llm_proxy_configured": bool(api_base and api_key),
        "api_base_url": api_base or "NOT CONFIGURED",
        "api_key_set": bool(api_key),
        "message": "LLM proxy ready for Phase 2 validation",
    })


@app.post("/run_inference", tags=["Testing"])
def run_inference():
    """Run LLM-powered inference (uses LLM proxy for Phase 2 validation)"""
    try:
        # Import and run the inference script that uses the LLM
        from inference import main as run_inference_main
        
        # This will use API_BASE_URL and API_KEY from environment
        run_inference_main()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Inference completed - LLM API calls were made",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": str(e),
            "message": "Inference failed"
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
