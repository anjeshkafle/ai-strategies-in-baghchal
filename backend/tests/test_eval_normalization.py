#!/usr/bin/env python3
import sys
import os
import json
from typing import List, Dict, Optional

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mcts_agent import MCTSAgent
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState

#-----------------------------------------------
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
#-----------------------------------------------

# Board states in an easy-to-edit format - add your test boards here
BOARD_STRING_1 = [
    "TGG_T",
    "G__TG",
    "G___G",
    "G___G",
    "TGGGT"
]

BOARD_STRING_2 = [
    "T___T",
    "__G__",
    "_____",
    "_____",
    "T___T"
]

BOARD_STRING_3 = [
    "T___T",
    "_____",
    "_____",
    "_____",
    "TG__T"
]

# Select which board to use (directly set to the board string variable)
BOARD_TO_USE = BOARD_STRING_1

# Configure game state settings
GAME_PHASE = "PLACEMENT"  # "PLACEMENT" or "MOVEMENT"
TURN = "GOAT"            # "GOAT" or "TIGER"
GOATS_PLACED = 1
GOATS_CAPTURED = 0

#-----------------------------------------------
# HELPER FUNCTIONS
#-----------------------------------------------

def string_board_to_game_state(string_board: List[str], phase=GAME_PHASE, turn=TURN, goats_placed=GOATS_PLACED, goats_captured=GOATS_CAPTURED):
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


def print_board(state: GameState) -> None:
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


def format_move(move: Optional[Dict]) -> str:
    """Format a move dictionary in a human-readable way."""
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


def test_normalization(game_state: GameState, mcts_agent: MCTSAgent, minimax_agent: MinimaxAgent):
    """
    Test the normalization function on all valid moves from the given state.
    Shows the raw evaluation scores and their normalized win rates.
    """
    print("\n" + "="*50)
    print("EVALUATING MOVES AND THEIR NORMALIZED SCORES")
    print("="*50)
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"Valid moves for {game_state.turn}: {len(valid_moves)}")
    
    # Get and sort moves by minimax evaluation
    move_evals = []
    for move in valid_moves:
        # Apply the move to a cloned state
        next_state = game_state.clone()
        next_state.apply_move(move)
        
        # Get the minimax evaluation
        eval_score = minimax_agent.evaluate(next_state)
        
        # Get the normalized win rate
        win_rate = mcts_agent._normalize_eval_score(eval_score)
        
        # Store the results
        move_evals.append((move, eval_score, win_rate))
    
    # Sort by evaluation score (ascending)
    move_evals.sort(key=lambda x: x[1])
    
    # Print all move evaluations and their normalized scores
    print("\nMove Evaluations (sorted by raw score):")
    print(f"{'Move':<30} {'Minimax Score':>15} {'Win Rate':>10}")
    print("-"*60)
    
    for move, eval_score, win_rate in move_evals:
        move_str = format_move(move)
        print(f"{move_str:<30} {eval_score:>15.2f} {win_rate:>10.4f}")
    
    # Print analysis based on player's perspective
    print("\nAnalysis from player's perspective:")
    if game_state.turn == "GOAT":
        print("  For GOAT: Lower win rates are better (closer to 0.0)")
        best_moves = sorted(move_evals, key=lambda x: x[2])[:3]
        print("\n  Top 3 moves for GOAT:")
        for move, eval_score, win_rate in best_moves:
            print(f"  - {format_move(move)}: Score = {eval_score:.2f}, Win Rate = {win_rate:.4f}")
    else:
        print("  For TIGER: Higher win rates are better (closer to 1.0)")
        best_moves = sorted(move_evals, key=lambda x: x[2], reverse=True)[:3]
        print("\n  Top 3 moves for TIGER:")
        for move, eval_score, win_rate in best_moves:
            print(f"  - {format_move(move)}: Score = {eval_score:.2f}, Win Rate = {win_rate:.4f}")
    
    # Score distribution analysis
    scores = [eval_score for _, eval_score, _ in move_evals]
    win_rates = [win_rate for _, _, win_rate in move_evals]
    
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        min_win_rate = min(win_rates)
        max_win_rate = max(win_rates)
        avg_win_rate = sum(win_rates) / len(win_rates)
        
        print("\nScore Distribution:")
        print(f"  Min Score: {min_score:.2f} -> Win Rate: {min_win_rate:.4f}")
        print(f"  Max Score: {max_score:.2f} -> Win Rate: {max_win_rate:.4f}")
        print(f"  Avg Score: {avg_score:.2f} -> Win Rate: {avg_win_rate:.4f}")
        print(f"  Score Range: {max_score - min_score:.2f}")
        print(f"  Win Rate Range: {max_win_rate - min_win_rate:.4f}")
        
        # Analyze if the normalization creates sufficient discrimination
        if max_win_rate - min_win_rate < 0.3:
            print("\nWARNING: Win rate range is small (<0.3).")
            print("Consider adjusting the normalization function for better discrimination.")
        else:
            print("\nWin rate range is sufficient for move discrimination.")


def main():
    """Run the normalization test with the specified board."""
    # Create a game state from the board
    game_state = string_board_to_game_state(BOARD_TO_USE, 
                                          phase=GAME_PHASE,
                                          turn=TURN,
                                          goats_placed=GOATS_PLACED,
                                          goats_captured=GOATS_CAPTURED)
    
    # Print the board state for verification
    print("\n" + "="*50)
    print("GAME STATE INFORMATION")
    print("="*50)
    print_board(game_state)
    
    # Print state information
    print(f"Turn: {game_state.turn}")
    print(f"Phase: {game_state.phase}")
    print(f"Goats placed: {game_state.goats_placed}")
    print(f"Goats captured: {game_state.goats_captured}")
    print(f"Game status: {game_state.game_status}")
    
    # Create the agents
    minimax_agent = MinimaxAgent()
    mcts_agent = MCTSAgent()
    
    # Test normalization
    test_normalization(game_state, mcts_agent, minimax_agent)
    
    # Provide instructions for tuning
    print("\n" + "="*50)
    print("HOW TO USE THIS TEST")
    print("="*50)
    print("1. Run this test with different board states by changing BOARD_TO_USE")
    print("2. Analyze the evaluation scores and their normalized win rates")
    print("3. Adjust the _normalize_eval_score function in mcts_agent.py")
    print("4. Re-run the test to see the effects of your changes")
    print("5. Repeat until you find a normalization that creates good discrimination")


if __name__ == "__main__":
    main() 