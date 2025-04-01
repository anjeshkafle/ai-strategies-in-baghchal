#!/usr/bin/env python3
import sys
import os
from typing import List, Dict, Optional
import json
import time

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState

#-----------------------------------------------
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
#-----------------------------------------------

# Board states in an easy-to-edit format
BOARD_STRING_1 = [
    "T_GG_",
    "_GG__",
    "_G_TT",
    "_T_GG",
    "GGGGG"
]

BOARD_STRING_2 = [
    "T___T",
    "_____",
    "_____",
    "__G__",
    "T___T"
]

BOARD_STRING_3 = [
    "T___T",
    "_____",
    "_____",
    "__G__",
    "T__GT"
]

# Select which board to use (directly set to the board string variable)
BOARD_TO_USE = BOARD_STRING_2

# Configure game state settings
GAME_PHASE = "PLACEMENT"  # "PLACEMENT" or "MOVEMENT"
TURN = "TIGER"           # "GOAT" or "TIGER"
GOATS_PLACED = 15
GOATS_CAPTURED = 0

def string_board_to_game_state(string_board: List[str], phase=GAME_PHASE, turn=TURN, goats_placed=GOATS_PLACED, goats_captured=GOATS_CAPTURED):
    """
    Convert a string-based board representation to a GameState object.
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

def print_board(state: GameState) -> None:
    """
    Print the current board state in a readable format.
    """
    print("\nCurrent Board State:")
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

def test_move_ordering():
    """Test move ordering by displaying the results of MinimaxAgent's _order_moves method."""
    # Create a game state from the board
    game_state = string_board_to_game_state(
        BOARD_TO_USE,  # Using board 1 which has goats
        phase=GAME_PHASE,
        turn=TURN,
        goats_placed=GOATS_PLACED,
        goats_captured=GOATS_CAPTURED
    )
    
    # Print the initial board state
    print("\nInitial Board State:")
    print_board(game_state)
    print(f"\nTurn: {game_state.turn}")
    print(f"Phase: {game_state.phase}")
    print(f"Goats placed: {game_state.goats_placed}")
    print(f"Goats captured: {game_state.goats_captured}")
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"\nTotal valid moves: {len(valid_moves)}")
    
    # Create a MinimaxAgent and get ordered moves
    agent = MinimaxAgent(max_depth=3)
    ordered_moves = agent._order_moves(game_state, valid_moves)
    
    # Display the ordered moves
    print("\nOrdered moves from MinimaxAgent:")
    for i, move in enumerate(ordered_moves, 1):
        print(f"{i}. {format_move(move)}")
    
    # Display threatened nodes for context
    print("\nThreatened nodes (potential capture positions):")
    threatened_data = game_state.get_threatened_nodes()
    for x, y, landing_x, landing_y in threatened_data:
        print(f"  Goat at ({x}, {y}) can be captured by landing at ({landing_x}, {landing_y})")

if __name__ == "__main__":
    test_move_ordering() 