#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.game_state import GameState
from models.minimax_agent import MinimaxAgent
from game_logic import get_all_possible_moves


def string_board_to_game_state(string_board, phase="PLACEMENT", turn="TIGER", goats_placed=1, goats_captured=0):
    """Convert a string representation of a board to a GameState object."""
    board = []
    for y, row in enumerate(string_board):
        board_row = []
        for x, cell in enumerate(row):
            if cell == 'T':
                board_row.append({"type": "TIGER"})
            elif cell == 'G':
                board_row.append({"type": "GOAT"})
            else:
                board_row.append(None)
        board.append(board_row)
    
    game_state = GameState()
    game_state.board = board
    game_state.phase = phase
    game_state.turn = turn
    game_state.goats_placed = goats_placed
    game_state.goats_captured = goats_captured
    
    return game_state

def print_board(state):
    """Print a simple text representation of the board."""
    print("-" * 11)
    for y in range(len(state.board)):
        row = "|"
        for x in range(len(state.board[y])):
            cell = state.board[y][x]
            if cell is None:
                row += "_|"
            elif cell["type"] == "TIGER":
                row += "T|"
            else:  # GOAT
                row += "G|"
        print(row)
        print("-" * 11)


def main():
    """Test for heuristics including the new positional score and optimal spacing."""

    board_2 = [
  "GG_GG",
  "GGGGG",
  "GGGTG",
  "GTT_G",
  "GGTGG"
]
    
    # Board weights matrix for reference (same as in the agent)
    board_weights = [
        [6, 5, 10, 5, 6],
        [5, 11, 7, 11, 5],
        [10, 7, 5, 7, 10],
        [5, 11, 7, 11, 5],
        [6, 5, 10, 5, 6]
    ]
    
    # Print the weights matrix for reference
    print("Board Weights Matrix:")
    for row in board_weights:
        print(row)
    print()
    
    # Test all boards
    test_boards = {
        "Board 2 (1 closed space working)": board_2,
    }
    
    agent = MinimaxAgent(max_depth=3)
    
    for board_name, board in test_boards.items():
        print(f"\n===== Testing {board_name} =====")
        
        # Create game state
        game_state = string_board_to_game_state(
            board, 
            phase="MOVEMENT",
            turn="TIGER",
            goats_placed=20,
            goats_captured=0
        )
        
        # Print the board
        print_board(game_state)
        
        # Display threatened nodes analysis
        threatened_nodes = game_state.get_threatened_nodes()
        print("\n=== THREATENED NODES ANALYSIS ===")
        print(f"Total threatened nodes: {len(threatened_nodes)}")
        print("Threatened positions (x, y) -> landing position (landing_x, landing_y):")
        for x, y, landing_x, landing_y in threatened_nodes:
            is_landing_empty = game_state.board[landing_y][landing_x] is None
            landing_status = "empty" if is_landing_empty else "occupied"
            print(f"  ({x}, {y}) -> ({landing_x}, {landing_y}) [Landing: {landing_status}]")
        print("="*35)
        
        # Get all tiger moves (needed for multiple heuristics)
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        
        # Print key heuristic information
        movable_tigers = agent._count_movable_tigers(all_tiger_moves)
        closed_regions = agent._count_closed_spaces(game_state, all_tiger_moves)
        threatened_goats = agent._count_threatened_goats(all_tiger_moves, game_state.turn)
        
        # Print detailed information about each closed region
        print("\nDetailed Closed Regions:")
        for i, region in enumerate(closed_regions):
            print(f"Region {i+1}: {region}")
        
        # Calculate tiger positional score (normalized 0-1 score)
        position_score = agent._calculate_tiger_positional_score(game_state)
        position_weight = agent.dispersion_weight  # Same weight as in evaluate function
        weighted_position = position_weight * position_score
        
        # Calculate tiger optimal spacing score (normalized 0-1 score)
        optimal_spacing_score = agent._calculate_tiger_optimal_spacing(game_state)
        optimal_spacing_weight = int(agent.dispersion_weight * 1.5)  # 50% more weight than positional score
        weighted_spacing = optimal_spacing_weight * optimal_spacing_score
        
        # Calculate goat edge preference (normalized 0-1 score)
        edge_score = agent._calculate_goat_edge_preference(game_state)
        edge_weight = agent.edge_weight  # Same weight as in evaluate function
        weighted_edge = edge_weight * edge_score
        
        print(f"\nHeuristic Results:")
        print(f"Movable Tigers: {movable_tigers}")
        print(f"Threatened Goats: {threatened_goats}")
        print(f"Closed Regions: {len(closed_regions)}")
        print(f"Total Closed Positions: {sum(len(region) for region in closed_regions)}")
        
        print(f"\nNew Heuristics:")
        print(f"1. Tiger Positional Score: {position_score:.3f} (normalized 0-1)")
        print(f"   Weighted Position Score: {weighted_position:.1f} (weight: {position_weight})")
        print(f"2. Tiger Optimal Spacing: {optimal_spacing_score:.3f} (normalized 0-1)")
        print(f"   Weighted Optimal Spacing: {weighted_spacing:.1f} (weight: {optimal_spacing_weight})")
        print(f"3. Goat Edge Preference: {edge_score:.3f} (normalized 0-1)")
        print(f"   Weighted Edge Preference: {weighted_edge:.1f} (weight: {edge_weight})")
        
        # Calculate the evaluation score
        eval_score = agent.evaluate(game_state)
        print(f"\nOverall Evaluation Score: {eval_score}")
        
        # Print tiger positions and their position values
        print("\nTiger Position Values:")
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if game_state.board[y][x] is not None and game_state.board[y][x]["type"] == "TIGER":
                    print(f"Tiger at ({x}, {y}) with position value: {board_weights[y][x]}")


if __name__ == "__main__":
    main() 