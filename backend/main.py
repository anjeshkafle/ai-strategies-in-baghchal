from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from models.random_agent import RandomAgent

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    board: List[List[Optional[Dict]]]
    phase: str
    agent: str
    model: str = "random"

# Initialize agents
agents = {
    "random": RandomAgent()
}

@app.post("/get-best-move")
async def get_best_move(request: MoveRequest):
    if request.model not in agents:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    
    return agents[request.model].get_move(request.board, request.phase, request.agent)

@app.get("/")
async def root():
    return {"message": "Welcome to the Bagh Chal AI Backend"}

# Add endpoints for AI move calculations here