"""
app.py - FastAPI web server for BBMP Road Repair Environment.
Exposes reset(), step(), state() as HTTP endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import BBMPAction, StepResult, ResetResult, StateResult
from environment import BBMPEnvironment

app = FastAPI(
    title="BBMP Road Repair Prioritization Environment",
    description="OpenEnv environment for training AI agents to prioritize road repairs in Bengaluru",
    version="1.0.0"
)

# One environment instance per task
envs = {
    "task1": BBMPEnvironment("task1"),
    "task2": BBMPEnvironment("task2"),
    "task3": BBMPEnvironment("task3"),
}

@app.get("/")
def root():
    return {
        "name": "BBMP Road Repair Environment",
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3"],
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: str = "task1"):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")
    result = envs[task_id].reset()
    return result.model_dump()

@app.post("/step")
def step(action: BBMPAction, task_id: str = "task1"):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")
    result = envs[task_id].step(action)
    return result.model_dump()

@app.get("/state")
def state(task_id: str = "task1"):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")
    result = envs[task_id].state()
    return result.model_dump()