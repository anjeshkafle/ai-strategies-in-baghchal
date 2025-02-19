from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from typing import Dict, Any, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Visual board representation (matches what you see on the actual board)
# Reading from top to bottom, left to right:
#   - Each string represents a column
#   - First string is leftmost column, last string is rightmost column
#   - Within each string, first char is top position, last char is bottom position
test_state = {
    "board": ["TGGGG", "GGGGG", "GGGTG", "GGT.G", "GG..T"],  # Visually intuitive format
    "phase": "PLACEMENT",
    "turn": "GOAT",
    "goats_placed": 18,
    "goats_captured": 0
}

def transpose_board(board_strings: List[str]) -> List[str]:
    """
    Transpose the visual board representation to match the internal coordinate system.
    Input format: Each string represents a column (visual format)
    Output format: Each string represents a row (internal format)
    """
    # Convert list of strings to list of lists for easier transposition
    char_lists = [list(s) for s in board_strings]
    # Transpose the 2D array
    transposed = list(map(list, zip(*char_lists)))
    # Convert back to list of strings
    return [''.join(row) for row in transposed]

def convert_compact_board(compact_board: List[str]) -> List[List[Dict]]:
    """
    Convert compact board representation to full board representation.
    Automatically transposes the board from visual format to internal format.
    """
    # First transpose the board to match internal coordinate system
    internal_board = transpose_board(compact_board)
    
    # Then convert to full board representation
    board = []
    for row in internal_board:
        board_row = []
        for cell in row:
            if cell == 'T':
                board_row.append({"type": "TIGER"})
            elif cell == 'G':
                board_row.append({"type": "GOAT"})
            else:  # cell == '.'
                board_row.append(None)
        board.append(board_row)
    return board

def create_game_state_from_dict(state_dict: Dict[str, Any]) -> GameState:
    """Create a GameState instance from a dictionary representation."""
    game_state = GameState()
    # Convert the compact board representation to full board representation
    game_state.board = convert_compact_board(state_dict["board"])
    game_state.phase = state_dict["phase"]
    game_state.turn = state_dict["turn"]
    game_state.goats_placed = state_dict["goats_placed"]
    game_state.goats_captured = state_dict["goats_captured"]
    return game_state

def test_minimax_agent():
    """Test the minimax agent on the given board state."""
    # Create game state from test state
    game_state = create_game_state_from_dict(test_state)
    
    # Initialize minimax agent with depth 4
    agent = MinimaxAgent(max_depth=4)
    
    # Get the best move
    best_move = agent.get_move(game_state)
    
    # Print initial evaluation
    print(f"Initial position evaluation: {agent.evaluate(game_state)}")
    
    # Print the best move found
    print("\nBest move found:")
    print(json.dumps(best_move, indent=2))
    
    # Apply the move to see resulting position
    game_state.apply_move(best_move)
    print(f"\nEvaluation after move: {agent.evaluate(game_state)}")
    
    return best_move

if __name__ == "__main__":
    test_minimax_agent()