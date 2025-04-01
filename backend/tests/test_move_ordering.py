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
    "TG__T",
    "_____",
    "_____",
    "_____",
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
BOARD_TO_USE = BOARD_STRING_3

# Configure game state settings
GAME_PHASE = "MOVEMENT"  # "PLACEMENT" or "MOVEMENT"
TURN = "GOAT"           # "GOAT" or "TIGER"
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
    """
    Test and analyze move ordering for both tigers and goats.
    Creates a game state from the configured board and logs detailed information
    about how moves are ordered by the MinimaxAgent.
    """
    print("\n" + "="*50)
    print("TESTING MOVE ORDERING")
    print("="*50)
    
    # Create game state from board
    game_state = string_board_to_game_state(
        BOARD_TO_USE,
        phase=GAME_PHASE,
        turn=TURN,
        goats_placed=GOATS_PLACED,
        goats_captured=GOATS_CAPTURED
    )
    
    # Print initial board state
    print_board(game_state)
    print(f"\nTurn: {game_state.turn}")
    print(f"Phase: {game_state.phase}")
    print(f"Goats placed: {game_state.goats_placed}")
    print(f"Goats captured: {game_state.goats_captured}")
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"\nTotal valid moves: {len(valid_moves)}")
    
    # Create MinimaxAgent
    agent = MinimaxAgent(max_depth=5)
    
    # Get threatened nodes for analysis
    threatened_data = game_state.get_threatened_nodes()
    print("\nThreatened nodes:")
    for x, y, landing_x, landing_y in threatened_data:
        print(f"  Position ({x}, {y}) can be captured to ({landing_x}, {landing_y})")
    
    # Order moves
    ordered_moves = agent._order_moves(game_state, valid_moves)
    
    # Analyze and print move categories
    if game_state.turn == "TIGER":
        # For tigers, analyze captures vs non-captures
        capture_moves = [m for m in ordered_moves if m.get("capture")]
        other_moves = [m for m in ordered_moves if not m.get("capture")]
        
        print("\nMove Ordering Analysis (Tiger):")
        print(f"Capture moves ({len(capture_moves)}):")
        for move in capture_moves:
            print(f"  {format_move(move)}")
        
        print(f"\nOther moves ({len(other_moves)}):")
        for move in other_moves:
            print(f"  {format_move(move)}")
    
    else:  # GOAT turn
        # For goats, analyze moves by category
        hot_squares = {}  # key: (x, y), value: list of (landing_x, landing_y)
        landing_squares = {}  # key: (landing_x, landing_y), value: list of (x, y)
        
        for x, y, landing_x, landing_y in threatened_data:
            if (x, y) not in hot_squares:
                hot_squares[(x, y)] = []
            hot_squares[(x, y)].append((landing_x, landing_y))
            
            if (landing_x, landing_y) not in landing_squares:
                landing_squares[(landing_x, landing_y)] = []
            landing_squares[(landing_x, landing_y)].append((x, y))
        
        # Categorize moves
        safe_moves = []
        escape_moves = []
        blocking_moves = []
        unsafe_moves = []
        
        for move in ordered_moves:
            if game_state.phase == "PLACEMENT":
                target_x, target_y = move["x"], move["y"]
                
                # Check if placing on a hot square
                if (target_x, target_y) in hot_squares:
                    is_unsafe = False
                    for landing_x, landing_y in hot_squares[(target_x, target_y)]:
                        if game_state.board[landing_y][landing_x] is None:
                            unsafe_moves.append(move)
                            is_unsafe = True
                            break
                    if is_unsafe:
                        continue
                
                # Check if blocking a capture
                if (target_x, target_y) in landing_squares:
                    has_real_block = False
                    for hot_x, hot_y in landing_squares[(target_x, target_y)]:
                        if game_state.board[hot_y][hot_x] is not None and game_state.board[hot_y][hot_x]["type"] == "GOAT":
                            has_real_block = True
                            break
                    if has_real_block:
                        blocking_moves.append(move)
                        continue
                
                safe_moves.append(move)
                
            else:  # MOVEMENT phase
                from_x, from_y = move["from"]["x"], move["from"]["y"]
                to_x, to_y = move["to"]["x"], move["to"]["y"]
                
                # Check if escaping from a hot square
                if (from_x, from_y) in hot_squares:
                    has_real_threat = False
                    for landing_x, landing_y in hot_squares[(from_x, from_y)]:
                        if game_state.board[landing_y][landing_x] is None:
                            has_real_threat = True
                            break
                    if has_real_threat:
                        escape_moves.append(move)
                        continue
                
                # Check if moving to a hot square
                if (to_x, to_y) in hot_squares:
                    is_unsafe = False
                    for landing_x, landing_y in hot_squares[(to_x, to_y)]:
                        is_landing_empty = game_state.board[landing_y][landing_x] is None
                        goat_from_landing = (from_x, from_y) == (landing_x, landing_y)
                        if is_landing_empty or goat_from_landing:
                            unsafe_moves.append(move)
                            is_unsafe = True
                            break
                    if is_unsafe:
                        continue
                
                # Check if blocking a capture
                if (to_x, to_y) in landing_squares:
                    has_real_block = False
                    for hot_x, hot_y in landing_squares[(to_x, to_y)]:
                        if game_state.board[hot_y][hot_x] is not None and game_state.board[hot_y][hot_x]["type"] == "GOAT":
                            if (hot_x, hot_y) != (from_x, from_y):
                                has_real_block = True
                                break
                    if has_real_block:
                        blocking_moves.append(move)
                        continue
                
                safe_moves.append(move)
        
        print("\nMove Ordering Analysis (Goat):")
        print(f"Safe moves ({len(safe_moves)}):")
        for move in safe_moves:
            print(f"  {format_move(move)}")
        
        print(f"\nEscape moves ({len(escape_moves)}):")
        for move in escape_moves:
            print(f"  {format_move(move)}")
        
        print(f"\nBlocking moves ({len(blocking_moves)}):")
        for move in blocking_moves:
            print(f"  {format_move(move)}")
        
        print(f"\nUnsafe moves ({len(unsafe_moves)}):")
        for move in unsafe_moves:
            print(f"  {format_move(move)}")
    
    # Print final ordered moves
    print("\nFinal ordered moves:")
    for i, move in enumerate(ordered_moves):
        print(f"  {i+1}. {format_move(move)}")

if __name__ == "__main__":
    test_move_ordering() 