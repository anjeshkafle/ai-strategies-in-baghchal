#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from game_logic import get_all_possible_moves


def string_board_to_game_state(string_board, phase="PLACEMENT", turn="TIGER", goats_placed=1, goats_captured=0):
    """Convert a string-based board representation to a GameState object."""
    # Create an empty board
    board = [[None for _ in range(5)] for _ in range(5)]
    
    # Fill the board based on the string representation
    for y, row in enumerate(string_board):
        for x, cell in enumerate(row):
            if cell == 'T':
                board[y][x] = {"type": "TIGER"}
            elif cell == 'G':
                board[y][x] = {"type": "GOAT"}
    
    # Create the game state
    state = GameState()
    state.board = board
    state.phase = phase
    state.turn = turn
    state.goats_placed = goats_placed
    state.goats_captured = goats_captured
    
    return state


def print_board(state):
    """Print the current board state in a readable format."""
    print("Current Board State:")
    print("-" * 11)
    for y in range(GameState.BOARD_SIZE):
        row = "|"
        for x in range(GameState.BOARD_SIZE):
            piece = state.board[y][x]
            if piece is None:
                row += " |"
            elif piece["type"] == "TIGER":
                row += "T|"
            else:  # GOAT
                row += "G|"
        print(row)
        print("-" * 11)


def main():
    """Test for heuristics including the new positional score and optimal spacing."""
    # Define board states for testing
    # Board 1: Tigers clustered in corner
    board_1 = [
        "_____",
        "_T__T",
        "_____",
        "_____",
        "_T__T"
    ]
    
    # Board 2: Tigers distributed with some goats
    board_2 = [
        "____G",
        "_TGG_",
        "____G",
        "_____",
        "T__TT"
    ]
    
    # Board 3: Tigers positioned for optimal spacing (3 nodes apart / 2 empty between)
    # This should score well on the new optimal spacing heuristic
    board_3 = [
        "T_T__",
        "_____",
        "T_T__",
        "_____",
        "_____"
    ]
    
    # Board 4: Tigers on high-value positions based on the weights matrix
    # Should score well on positional score
    board_4 = [
        "_____",
        "_T_T_",
        "_____",
        "_T_T_",
        "_____"
    ]
    
    # A board to demonstrate threatened goats
    board_with_threats = [
        "T___T",
        "_G___",
        "_G_G_",
        "__G__",
        "T___T"
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
        "Board 1 (Tigers Clustered)": board_1,
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
        
        # Get all tiger moves (needed for multiple heuristics)
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        
        # Print key heuristic information
        movable_tigers = agent._count_movable_tigers(all_tiger_moves)
        closed_regions = agent._count_closed_spaces(game_state, all_tiger_moves)
        threatened_goats = agent._count_threatened_goats(all_tiger_moves)
        
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