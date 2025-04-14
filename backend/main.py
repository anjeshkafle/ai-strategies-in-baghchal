from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models.random_agent import RandomAgent
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from models.mcts_agent import MCTSAgent
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

class AgentSettings(BaseModel):
    # Shared settings
    randomize_equal_moves: Optional[bool] = None
    
    # Minimax specific settings
    max_depth: Optional[int] = Field(None, ge=1, le=9, description="Maximum search depth for Minimax (1-9)")
    
    # MCTS specific settings
    iterations: Optional[int] = Field(None, ge=100, le=100000, description="Number of iterations for MCTS")
    rollout_policy: Optional[str] = Field(None, description="Rollout policy ('random', 'guided', or 'lightweight')")
    max_rollout_depth: Optional[int] = Field(None, ge=1, le=20, description="Maximum rollout depth for MCTS")
    max_time_seconds: Optional[int] = Field(None, ge=1, le=300, description="Maximum time in seconds for MCTS")
    exploration_weight: Optional[float] = Field(None, ge=0.1, le=3.0, description="Exploration weight for MCTS")
    guided_strictness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Guided rollout strictness (0.0-1.0)")

class MoveRequest(BaseModel):
    board: List[List[Optional[Dict]]]
    phase: str
    agent: str
    model: str
    goats_placed: int = 0  # Add default value
    goats_captured: int = 0  # Add default value
    settings: Optional[AgentSettings] = None  # Optional agent settings

# Default agent settings
default_settings = {
    "minimax": {
        "max_depth": 6,
        "randomize_equal_moves": True
    },
    "mcts": {
        "iterations": 20000,
        "exploration_weight": 1.414,
        "rollout_policy": "lightweight",
        "guided_strictness": 0.7,
        "max_rollout_depth": 6,
        "max_time_seconds": 50
    }
}

# Initialize agents with default settings
agents = {
    "random": RandomAgent(),
    "minimax": MinimaxAgent(max_depth=default_settings["minimax"]["max_depth"], 
                           randomize_equal_moves=default_settings["minimax"]["randomize_equal_moves"]),
    "mcts": MCTSAgent(
        iterations=default_settings["mcts"]["iterations"], 
        exploration_weight=default_settings["mcts"]["exploration_weight"], 
        rollout_policy=default_settings["mcts"]["rollout_policy"],
        guided_strictness=default_settings["mcts"]["guided_strictness"],
        max_rollout_depth=default_settings["mcts"]["max_rollout_depth"],
        max_time_seconds=default_settings["mcts"]["max_time_seconds"]
    )
}

@app.get("/agent-settings/{model}")
async def get_agent_settings(model: str):
    """Get default settings for a specific agent model"""
    if model not in default_settings:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    return default_settings[model]

@app.get("/agent-settings")
async def get_all_agent_settings():
    """Get default settings for all agent models"""
    return default_settings

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
        
        # Get move using the agent, possibly with custom settings
        try:
            logger.info(f"Starting {request.model} agent calculation...")
            move_start_time = time.time()
            
            # Check if custom settings are provided
            if request.settings:
                # Create a new instance of the agent with custom settings
                if request.model == "minimax":
                    # Only update provided settings
                    custom_max_depth = request.settings.max_depth if request.settings.max_depth is not None else default_settings["minimax"]["max_depth"]
                    custom_randomize = request.settings.randomize_equal_moves if request.settings.randomize_equal_moves is not None else default_settings["minimax"]["randomize_equal_moves"]
                    
                    logger.info(f"Using custom minimax settings: max_depth={custom_max_depth}, randomize={custom_randomize}")
                    custom_agent = MinimaxAgent(max_depth=custom_max_depth, randomize_equal_moves=custom_randomize)
                    move = custom_agent.get_move(state)
                
                elif request.model == "mcts":
                    # Extract custom settings or use defaults
                    custom_iterations = request.settings.iterations if request.settings.iterations is not None else default_settings["mcts"]["iterations"]
                    custom_exploration = request.settings.exploration_weight if request.settings.exploration_weight is not None else default_settings["mcts"]["exploration_weight"]
                    custom_rollout = request.settings.rollout_policy if request.settings.rollout_policy is not None else default_settings["mcts"]["rollout_policy"]
                    custom_max_depth = request.settings.max_rollout_depth if request.settings.max_rollout_depth is not None else default_settings["mcts"]["max_rollout_depth"]
                    custom_max_time = request.settings.max_time_seconds if request.settings.max_time_seconds is not None else default_settings["mcts"]["max_time_seconds"]
                    custom_strictness = request.settings.guided_strictness if request.settings.guided_strictness is not None else default_settings["mcts"]["guided_strictness"]
                    
                    logger.info(f"Using custom MCTS settings: iterations={custom_iterations}, rollout={custom_rollout}, max_depth={custom_max_depth}, max_time_seconds={custom_max_time}")
                    custom_agent = MCTSAgent(
                        iterations=custom_iterations,
                        exploration_weight=custom_exploration,
                        rollout_policy=custom_rollout,
                        max_rollout_depth=custom_max_depth,
                        guided_strictness=custom_strictness,
                        max_time_seconds=custom_max_time
                    )
                    move = custom_agent.get_move(state)
                
                else:  # For random or any other agent without custom settings
                    move = agents[request.model].get_move(state)
            
            else:
                # Use default agents
                move = agents[request.model].get_move(state)
            
            move_elapsed = time.time() - move_start_time
            
            elapsed = time.time() - start_time
            logger.info(f"Agent calculation completed in {move_elapsed:.2f}s (total request: {elapsed:.2f}s)")
            
            # Log the selected move concisely
            if move:
                if "type" in move and move["type"] == "placement":
                    logger.info(f"Selected: Place at ({move['x']}, {move['y']}) [{elapsed:.2f}s]")
                else:
                    capture_info = " with capture" if move.get("capture") else ""
                    logger.info(f"Selected: ({move['from']['x']}, {move['from']['y']}) -> ({move['to']['x']}, {move['to']['y']}){capture_info} [{elapsed:.2f}s]")
            else:
                logger.warning(f"Agent returned no move after {move_elapsed:.2f}s")
            
            return move
        except Exception as e:
            logger.error(f"Error in agent.get_move(): {str(e)}")
            logger.exception("Detailed agent error traceback:")
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in request processing: {str(e)}")
        logger.exception("Detailed error traceback:")
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