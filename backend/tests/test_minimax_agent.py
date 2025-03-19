#!/usr/bin/env python3
import sys
import os
from typing import List, Dict, Optional
import json

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState


def string_board_to_game_state(string_board: List[str], phase="PLACEMENT", turn="TIGER", goats_placed=1, goats_captured=0):
    """
    Convert a string-based board representation to a GameState object.
    
    Args:
        string_board: A list of strings representing the board state.
        phase: The game phase (default: "PLACEMENT").
        turn: Whose turn it is (default: "TIGER").
        goats_placed: Number of goats placed (default: 1).
        goats_captured: Number of goats captured (default: 0).
        
    Returns:
        A GameState object initialized with the given board and settings.
    """
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


def print_board(state: GameState) -> None:
    """
    Print the current board state in a readable format.
    """
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


def format_move(move: Optional[Dict]) -> str:
    """
    Format a move dictionary in a human-readable way.
    """
    if move is None:
        return "No valid moves available"
    
    if move["type"] == "placement":
        return f"Place goat at ({move['x']}, {move['y']})"
    else:  # movement
        from_x, from_y = move["from"]["x"], move["from"]["y"]
        to_x, to_y = move["to"]["x"], move["to"]["y"]
        
        if move.get("capture"):
            cap_x, cap_y = move["capture"]["x"], move["capture"]["y"]
            return f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y}), capturing goat at ({cap_x}, {cap_y})"
        else:
            return f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y})"


def main():
    """Run a minimax agent test with configurable parameters."""
    # The board state in an easy-to-edit format
    string_board = [
        "T___T",
        "_____",
        "_____",
        "____G",
        "T___T"
    ]
    
    test_state_2 = [
      "_____",
      "_TGG_",
      "_____",
      "_____",
      "T__TT"
    ]

    test_state_3 = [
      "T___T",
      "_____",
      "___G_",
      "_____",
      "T___T"
    ]
    
    # Create a game state from the board
    # You can modify these parameters as needed
    game_state = string_board_to_game_state(
        test_state_2, 
        phase="PLACEMENT",
        turn="TIGER",
        goats_placed=1,
        goats_captured=0
    )
    
    # Print the board state for verification
    print_board(game_state)
    
    # Print state information
    print(f"Turn: {game_state.turn}")
    print(f"Phase: {game_state.phase}")
    print(f"Goats placed: {game_state.goats_placed}")
    print(f"Goats captured: {game_state.goats_captured}")
    print(f"Game status: {game_state.game_status}")
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"\nValid moves for {game_state.turn}: {len(valid_moves)}")
    
    # Get capture moves
    capture_moves = [move for move in valid_moves if move.get("capture")]
    print(f"Capture moves available: {len(capture_moves)}")
    
    if capture_moves:
        print("\nCapture moves:")
        for i, move in enumerate(capture_moves):
            print(f"  {i+1}. {format_move(move)}")
    
    # Create the minimax agent with depth=4
    agent = MinimaxAgent(max_depth=4)
    
    # Get the best move
    best_move = agent.get_move(game_state)
    
    # Print the best move
    print("\nBest move according to Minimax agent:")
    if best_move:
        print(format_move(best_move))
        print(f"Move score: {agent.best_score}")
        
        # Apply the move to see the result
        new_state = game_state.clone()
        new_state.apply_move(best_move)
        print("\nBoard state after applying the best move:")
        print_board(new_state)
        
        # Check if the move was a capture move
        if best_move.get("capture"):
            print("✓ The agent chose a capture move!")
        else:
            if capture_moves:
                print("✗ The agent did NOT choose a capture move, even though captures are available.")
            else:
                print("(No capture moves were available)")
    else:
        print("No move was returned by the agent.")


if __name__ == "__main__":
    main() 