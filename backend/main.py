from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from models.random_agent import RandomAgent
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    model: str
    goats_placed: int = 0  # Add default value
    goats_captured: int = 0  # Add default value

# Initialize agents
agents = {
    "random": RandomAgent(),
    "minimax": MinimaxAgent(max_depth=4)  # Remove max_time as it's not used
}

@app.post("/get-best-move")
async def get_best_move(request: MoveRequest):
    start_time = time.time()
    logger.info(f"Received move request for {request.model} model, {request.agent} agent in {request.phase} phase")
    
    if request.model not in agents:
        logger.error(f"Unknown model requested: {request.model}")
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    
    try:
        # Create a GameState object from the request
        state = GameState()
        state.board = request.board
        state.phase = request.phase
        state.turn = request.agent
        state.goats_placed = request.goats_placed
        state.goats_captured = request.goats_captured
        
        # Get move using the agent
        move = agents[request.model].get_move(state)
        
        elapsed = time.time() - start_time
        logger.info(f"Move calculation completed in {elapsed:.2f}s: {move}")
        return move
    except Exception as e:
        logger.error(f"Error calculating move: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Bagh Chal AI Backend"}

# Log startup
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete")

# Add endpoints for AI move calculations here