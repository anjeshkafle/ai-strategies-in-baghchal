from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
import json  # Add this for pretty printing

def convert_string_to_board(board_state):
    """Convert string representation to game board.
    Each string in the list represents a row from top to bottom.
    T = Tiger, G = Goat, _ = Empty
    
    Note: The internal board representation uses [y][x] indexing,
    so we need to swap coordinates when converting from visual representation.
    """
    board = [[None for _ in range(5)] for _ in range(5)]
    for row_idx, row in enumerate(board_state):
        for col_idx, cell in enumerate(row):
            if cell == 'T':
                board[row_idx][col_idx] = {"type": "TIGER"}  # Don't swap - GameState uses [row][col]
            elif cell == 'G':
                board[row_idx][col_idx] = {"type": "GOAT"}
            # _ represents empty cell, already None by default
    return board

def print_region_details(closed_regions):
    """Helper function to print detailed information about closed regions"""
    if not closed_regions:
        print("No closed regions found.")
        return
        
    print(f"\nFound {len(closed_regions)} closed region(s):")
    for i, region in enumerate(closed_regions):
        print(f"\nRegion {i+1} (size: {len(region)}):")
        print("Coordinates (x,y):", ", ".join([f"({x},{y})" for x,y in region]))
        
        # Calculate region bounds
        min_x = min(x for x,_ in region)
        max_x = max(x for x,_ in region)
        min_y = min(y for _,y in region)
        max_y = max(y for _,y in region)
        print(f"Region bounds: x:[{min_x},{max_x}], y:[{min_y},{max_y}]")
        
        # Visual representation of this region
        print("Visual shape:")
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                if (x,y) in region:
                    row += "□ "  # Empty square in region
                else:
                    row += "· "  # Not part of region
            print(row)

def test_count_functions(depth=4):
    # Test state with multiple closed regions
    test_state_1 = [
  "T____",
  "__GT_",
  "_____",
  "_____",
  "T___T"
]
    
    print("\n" + "="*50)
    print("TESTING BOARD STATE")
    print("="*50)
    
    game_state_1 = GameState()
    game_state_1.board = convert_string_to_board(test_state_1)
    game_state_1.goats_placed = 4  # Set to placement phase
    game_state_1.phase = "PLACEMENT"
    game_state_1.turn = "GOAT"  # Explicitly set turn
    
    # Log the internal board representation
    print("\nInternal board representation:")
    for row in game_state_1.board:
        print(json.dumps(row))
    
    # Create minimax agent with specified depth
    agent = MinimaxAgent(max_depth=depth)
    
    # Count movable tigers
    movable_tigers = agent._count_movable_tigers(game_state_1)
    print(f"\nNumber of movable tigers: {movable_tigers}")
    
    # Get and analyze closed regions
    closed_regions = agent._count_closed_spaces(game_state_1)
    total_closed_spaces = sum(len(region) for region in closed_regions)
    print(f"\nTotal closed spaces: {total_closed_spaces}")
    print_region_details(closed_regions)
    
    # Get best move from minimax agent
    best_move = agent.get_move(game_state_1)
    print(f"\nBest move found (depth={depth}):")
    print(json.dumps(best_move, indent=2))
    print(f"Evaluation score: {agent.evaluate(game_state_1)}")
    
    # Print the board for visualization
    print("\nComplete board state (visual):")
    for i in range(5):
        row = ""
        for j in range(5):
            piece = game_state_1.board[i][j]  # Don't swap - we want visual representation
            if piece is None:
                row += "_ "
            elif piece["type"] == "TIGER":
                row += "T "
            else:
                row += "G "
        print(row.strip())
    
    print(f"\nCurrent turn: {game_state_1.turn}")
    print(f"Phase: {game_state_1.phase}")
    print(f"Goats placed: {game_state_1.goats_placed}")

if __name__ == "__main__":
    test_count_functions()
    
# Add a pytest-compatible test function
def test_minimax_agent():
    """Pytest-compatible test function for the minimax agent."""
    # Test with a smaller depth for faster execution during automated testing
    test_count_functions(depth=3)
    # No assertions needed as this is primarily a visual test
    # The test passes if no exceptions are raised