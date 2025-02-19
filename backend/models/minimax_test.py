from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from game_logic import get_all_possible_moves
from typing import Dict, Any, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_board_state(state: GameState):
    """
    Print a visual representation of the board state.
    Transposes the board back to original orientation where:
    - Each row is printed from top to bottom
    - Within each row, pieces are printed from left to right
    """
    print("\nBoard state:")
    # Transpose back for display
    display_board = [[None for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            display_board[i][j] = state.board[j][i]
    
    for row in display_board:
        row_str = ""
        for piece in row:
            if piece is None:
                row_str += ". "
            elif piece["type"] == "TIGER":
                row_str += "T "
            else:
                row_str += "G "
        print(row_str.rstrip())
    print(f"Turn: {state.turn}, Phase: {state.phase}")
    print(f"Goats placed: {state.goats_placed}, Goats captured: {state.goats_captured}\n")

# Visual board representation:
# In test_state["board"]:
#   - Each string represents a ROW (top to bottom)
#   - First string is top row, last string is bottom row
#   - Within each string, chars go from left to right
# Example:
# ["TGGGG",  # Top row
#  "GGGGG",  # Second row
#  "GGGTG",  # Middle row
#  "GGT.G",  # Fourth row
#  "GG..T"]  # Bottom row
#["GGG..", "T..T.", "...G.", ".....", "T...T"] //ALlows a tiger to capture a goat
#["T...T", ".....", ".....", ".....", "T...T"] //No goats captured
test_state = {
    "board": ["GGG..", "T..T.", "...G.", ".....", "T...T"],
    "phase": "PLACEMENT",
    "turn": "TIGER",
    "goats_placed": 4,
    "goats_captured": 0
}

def transpose_board(board_strings: List[str]) -> List[str]:
    """
    Convert from test_state format to internal format.
    
    Input (test_state format):
    - Each string represents a row
    - First string is top row
    Example: ["TGGGG",  # Top row
             "GGGGG",  # Second row
             "GGGTG",  # Middle row
             "GGT.G",  # Fourth row
             "GG..T"]  # Bottom row
    
    Output (internal format):
    - Each string represents a column
    - First string is leftmost column
    Example: ["TGGGG",  # Leftmost column
             "GGGGG",  # Second column
             "GGGTG",  # Middle column
             "GGT.G",  # Fourth column
             "GG..T"]  # Rightmost column
    """
    # No need to transpose since the format is already correct
    return board_strings

def convert_compact_board(compact_board: List[str]) -> List[List[Dict]]:
    """
    Convert compact board representation to full board representation.
    Transposes the board so that (x,y) in moves can be interpreted as (row,col).
    """
    # First create board in original orientation
    board = []
    for row in compact_board:  # Each string is a row
        board_row = []
        for cell in row:  # Each char in the string is a cell in that row
            if cell == 'T':
                board_row.append({"type": "TIGER"})
            elif cell == 'G':
                board_row.append({"type": "GOAT"})
            else:  # cell == '.'
                board_row.append(None)
        board.append(board_row)
    
    # Now transpose the board so (x,y) in moves can be interpreted as (row,col)
    transposed = [[None for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            transposed[j][i] = board[i][j]
    
    return transposed

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
    
    print("\nInitial board state:")
    debug_board_state(game_state)
    
    # Initialize minimax agent with depth 4
    agent = MinimaxAgent(max_depth=4)
    
    # Get the best move
    best_move = agent.get_move(game_state)
    
    # Print initial evaluation
    print(f"Initial position evaluation: {agent.evaluate(game_state)}")
    
    # Print the best move found
    print("\nBest move found:")
    print(json.dumps(best_move, indent=2))
    
    # Apply the move and show resulting state
    game_state.apply_move(best_move)
    print("\nAfter applying move:")
    debug_board_state(game_state)
    
    # Show possible tiger moves in the resulting position
    tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
    print("\nPossible tiger moves in resulting position:")
    for move in tiger_moves:
        print(json.dumps(move, indent=2))
    
    print(f"\nEvaluation after move: {agent.evaluate(game_state)}")
    
    return best_move

if __name__ == "__main__":
    test_minimax_agent()