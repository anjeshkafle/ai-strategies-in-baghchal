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
    """Simple test for movable tigers and closed spaces."""
    # Define the three board states from test_minimax_agent
    board_1 = [
        "T___T",
        "____G",
        "____G",
        "____G",
        "T___T"
    ]
    
    board_2 = [
      "____G",
      "_TGG_",
      "____G",
      "_____",
      "T__TT"
    ]

    test_state_3 = [
      "T_T__",
      "_____",
      "T_T__",
      "_____",
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
    
    # Select which board to test (change this variable to test different boards)
    selected_board = board_1
    
    # Create game state
    game_state = string_board_to_game_state(
        selected_board, 
        phase="MOVEMENT",
        turn="TIGER",
        goats_placed=20,
        goats_captured=0
    )
    
    # Print the board
    print_board(game_state)
    
    # Create the minimax agent
    agent = MinimaxAgent(max_depth=3)
    
    # Get all tiger moves first (needed for multiple heuristics)
    all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
    
    # Print key heuristic information
    movable_tigers = agent._count_movable_tigers(all_tiger_moves)
    closed_regions = agent._count_closed_spaces(game_state, all_tiger_moves)
    threatened_goats = agent._count_threatened_goats(all_tiger_moves)
    
    # Calculate tiger dispersion (normalized 0-1 score)
    dispersion_score = agent._calculate_tiger_dispersion(game_state)
    dispersion_weight = 100  # Same weight as in evaluate function
    weighted_dispersion = dispersion_weight * dispersion_score
    
    # Calculate goat edge preference (normalized 0-1 score)
    edge_score = agent._calculate_goat_edge_preference(game_state)
    edge_weight = 150  # Same weight as in evaluate function
    weighted_edge = edge_weight * edge_score
    
    print(f"\nMovable Tigers: {movable_tigers}")
    print(f"Threatened Goats: {threatened_goats}")
    print(f"Closed Regions: {len(closed_regions)}")
    print(f"Total Closed Positions: {sum(len(region) for region in closed_regions)}")
    print(f"Tiger Dispersion: {dispersion_score:.3f} (normalized 0-1)")
    print(f"Weighted Dispersion: {weighted_dispersion:.1f} (weight: {dispersion_weight})")
    print(f"Goat Edge Preference: {edge_score:.3f} (normalized 0-1)")
    print(f"Weighted Edge Preference: {weighted_edge:.1f} (weight: {edge_weight})")
    
    # Calculate the evaluation score
    eval_score = agent.evaluate(game_state)
    print(f"\nOverall Evaluation Score: {eval_score}")
    
    # Print regions if any exist
    if closed_regions:
        print("\nClosed Regions Details:")
        for i, region in enumerate(closed_regions):
            print(f"Region {i+1}: {region}")


if __name__ == "__main__":
    main() 