from typing import List, Dict, Optional, TypedDict, Union
from models.game_state import GameState
import random
from game_logic import get_all_possible_moves

class PlacementMove(TypedDict):
    type: str  # "placement"
    x: int
    y: int

class MovementMove(TypedDict):
    type: str  # "movement"
    from_: Dict[str, int]
    to: Dict[str, int]
    capture: Optional[Dict[str, int]]

class RandomAgent:
    def get_move(self, state: GameState) -> Union[PlacementMove, MovementMove]:
        """
        Get a random valid move for the current state
        Returns either:
        - Placement move: {"type": "placement", "x": int, "y": int}
        - Movement move: {
            "type": "movement",
            "from": {"x": int, "y": int},
            "to": {"x": int, "y": int},
            "capture": Optional[{"x": int, "y": int}]
          }
        """
        possible_moves = get_all_possible_moves(state.board, state.phase, state.turn)
        if not possible_moves:
            raise ValueError(f"No valid moves available for {state.turn} in current state")
        
        return random.choice(possible_moves) 