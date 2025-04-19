from typing import Dict, List, Tuple, Optional, Any
import time
import json
import uuid
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.game_state import GameState
from models.mcts_agent import MCTSAgent
from models.minimax_agent import MinimaxAgent

class GameRunner:
    """
    Runs a single game between two AI agents and captures the results.
    Handles threefold repetition detection and game statistics collection.
    """
    
    def __init__(self, tiger_config: Dict, goat_config: Dict):
        """
        Initialize a game runner with agent configurations.
        
        Args:
            tiger_config: Configuration dict for the Tiger agent
            goat_config: Configuration dict for the Goat agent
        """
        self.tiger_config = tiger_config
        self.goat_config = goat_config
        self.game_id = str(uuid.uuid4())  # Generate a unique game ID
        self.move_history = []
        
    def _create_agent(self, config: Dict) -> Any:
        """
        Create an agent based on configuration.
        
        Args:
            config: Agent configuration with at least 'algorithm' key
            
        Returns:
            An instance of the appropriate agent class
        """
        if config['algorithm'] == 'minimax':
            return MinimaxAgent(
                max_depth=config.get('max_depth', 3),
                randomize_equal_moves=config.get('randomize', True),
                useTunedParams=config.get('use_tuned_params', True)
            )
        elif config['algorithm'] == 'mcts':
            return MCTSAgent(
                iterations=config.get('iterations'),
                exploration_weight=config.get('exploration_weight', 1.0),
                rollout_policy=config.get('rollout_policy', 'random'),
                max_rollout_depth=config.get('rollout_depth', 6),
                guided_strictness=config.get('guided_strictness', 0.8),
                max_time_seconds=config.get('max_time_seconds', config.get('max_time', 30))
            )
        else:
            raise ValueError(f"Unknown algorithm: {config['algorithm']}")
            
    def _format_move(self, move: Dict) -> str:
        """
        Format a move into condensed notation.
        
        Args:
            move: Move dictionary
            
        Returns:
            Condensed string representation of the move
        """
        if move["type"] == "placement":
            return f"p{move['x']}{move['y']}"
        else:  # movement
            from_x, from_y = move["from"]["x"], move["from"]["y"]
            to_x, to_y = move["to"]["x"], move["to"]["y"]
            
            if move.get("capture"):
                cap_x, cap_y = move["capture"]["x"], move["capture"]["y"]
                return f"m{from_x}{from_y}{to_x}{to_y}c{cap_x}{cap_y}"
            else:
                return f"m{from_x}{from_y}{to_x}{to_y}"
    
    def _get_state_hash(self, state: GameState) -> str:
        """
        Create a hash representation of the game state for repetition detection.
        
        Args:
            state: Current game state
            
        Returns:
            String hash of the board state
        """
        # Only track repetition during movement phase
        if state.phase == "MOVEMENT":
            # Convert board to a string representation
            board_str = ""
            for row in state.board:
                for cell in row:
                    if cell is None:
                        board_str += "_"
                    elif cell["type"] == "TIGER":
                        board_str += "T"
                    else:
                        board_str += "G"
            
            # Include turn in the hash
            return f"{board_str}_{state.turn}"
        else:
            # During placement phase, include goats_placed to ensure uniqueness
            return f"PLACEMENT_{state.goats_placed}_{state.turn}"
    
    def run_game(self) -> Dict:
        """
        Run a complete game and return statistics.
        
        Returns:
            Dict containing game statistics
        """
        # Initialize game state
        state = GameState()
        
        # Create agents
        tiger_agent = self._create_agent(self.tiger_config)
        goat_agent = self._create_agent(self.goat_config)
        
        # Statistics
        start_time = time.time()
        move_count = 0
        tiger_times = []
        goat_times = []
        
        # Track visited states for threefold repetition
        visited_states = {}  # Format: {state_hash: count}
        
        # Main game loop
        while not state.is_terminal():
            # Get current player
            current_agent = tiger_agent if state.turn == "TIGER" else goat_agent
            
            # Check for threefold repetition in movement phase
            if state.phase == "MOVEMENT":
                state_hash = self._get_state_hash(state)
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    # Game ends in a draw due to threefold repetition
                    draw_result = {
                        "game_id": self.game_id,
                        "tiger_config": self.tiger_config,
                        "goat_config": self.goat_config,
                        "winner": "DRAW",
                        "reason": "THREEFOLD_REPETITION",
                        "moves": move_count,
                        "game_duration": time.time() - start_time,
                        "avg_tiger_move_time": sum(tiger_times) / len(tiger_times) if tiger_times else 0,
                        "avg_goat_move_time": sum(goat_times) / len(goat_times) if goat_times else 0,
                        "first_capture_move": next((i for i, m in enumerate(self.move_history) if "c" in m), None),
                        "goats_captured": state.goats_captured,
                        "phase_transition_move": next((i for i, m in enumerate(self.move_history) if state.goats_placed == 20), None),
                        "move_history": ",".join(self.move_history)
                    }
                    return draw_result
            
            # Measure move time
            move_start = time.time()
            
            # Get the agent's move
            move = current_agent.get_move(state)
            
            # Record move time
            move_time = time.time() - move_start
            if state.turn == "TIGER":
                tiger_times.append(move_time)
            else:
                goat_times.append(move_time)
            
            # Format and add move to history
            move_notation = self._format_move(move)
            self.move_history.append(move_notation)
            
            # Apply the move
            state.apply_move(move)
            move_count += 1
        
        # Game has ended with a winner
        game_result = {
            "game_id": self.game_id,
            "tiger_config": self.tiger_config,
            "goat_config": self.goat_config,
            "winner": state.get_winner(),
            "reason": "STANDARD",
            "moves": move_count,
            "game_duration": time.time() - start_time,
            "avg_tiger_move_time": sum(tiger_times) / len(tiger_times) if tiger_times else 0,
            "avg_goat_move_time": sum(goat_times) / len(goat_times) if goat_times else 0,
            "first_capture_move": next((i for i, m in enumerate(self.move_history) if "c" in m), None),
            "goats_captured": state.goats_captured,
            "phase_transition_move": next((i for i, m in enumerate(self.move_history) if i > 0 and self.move_history[i-1].startswith('p') and self.move_history[i].startswith('m')), None),
            "move_history": ",".join(self.move_history)
        }
        
        return game_result 