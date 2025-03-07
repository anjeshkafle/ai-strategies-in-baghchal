from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from models.random_agent import RandomAgent
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
import logging
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to capture more detailed information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    "minimax": MinimaxAgent(max_depth=6)  # Remove max_time as it's not used
}

@app.post("/get-best-move")
async def get_best_move(request: MoveRequest):
    start_time = time.time()
    
    # Log board state in a compact format
    logger.info("BOARD STATE:")
    for row in request.board:
        logger.info(" ".join([piece.get("type", "-")[0] if piece else "." for piece in row]))
    logger.info(f"Phase: {request.phase} | Turn: {request.agent} | Goats: placed={request.goats_placed}, captured={request.goats_captured}")
    
    # Log serialized board state for replay
    serialized = {
        "board": request.board,
        "phase": request.phase,
        "agent": request.agent,
        "model": request.model,
        "goats_placed": request.goats_placed,
        "goats_captured": request.goats_captured
    }
    logger.info(f"Serialized state: {json.dumps(serialized)}")
    
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
        
        # Log the selected move concisely
        if move:
            if "type" in move and move["type"] == "placement":
                logger.info(f"Selected: Place at ({move['x']}, {move['y']}) [{elapsed:.2f}s]")
            else:
                capture_info = " with capture" if move.get("capture") else ""
                logger.info(f"Selected: ({move['from']['x']}, {move['from']['y']}) -> ({move['to']['x']}, {move['to']['y']}){capture_info} [{elapsed:.2f}s]")
        
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