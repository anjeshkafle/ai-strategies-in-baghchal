#!/usr/bin/env python3
"""
Test script for Bagh Chal simulation system.

This script runs a single game between two AI agents with
specific configurations and prints detailed results to the console.

Simply edit the configuration settings at the top of the file
and run the script without any command line arguments.
"""

import os
import sys
import json
import pprint
import time
from typing import Dict

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.simulation.game_runner import GameRunner

#-----------------------------------------------
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
#-----------------------------------------------

# Tiger Agent Configuration
TIGER_CONFIG = {
    'algorithm': 'mcts',           # Choose 'minimax' or 'mcts'
    
    # MCTS specific settings (only used if algorithm is 'mcts')
    'iterations': 10000,           # Number of MCTS iterations
    'rollout_policy': 'lightweight', # 'random', 'lightweight', or 'guided'
    'rollout_depth': 6,            # Maximum depth for rollouts
    'exploration_weight': 1.0,     # Exploration weight for UCB formula
    'guided_strictness': 0.8,      # Strictness for guided rollouts (0.0-1.0)
    'max_time': 60,                # Maximum time in seconds
    
    # Minimax specific settings (only used if algorithm is 'minimax')
    'depth': 5,                    # Search depth for Minimax
    'randomize': True              # Whether to randomize equal moves
}

# Goat Agent Configuration
GOAT_CONFIG = {
    'algorithm': 'minimax',        # Choose 'minimax' or 'mcts'
    
    # MCTS specific settings (only used if algorithm is 'mcts')
    'iterations': 15000,           # Number of MCTS iterations
    'rollout_policy': 'guided',    # 'random', 'lightweight', or 'guided'
    'rollout_depth': 4,            # Maximum depth for rollouts
    'exploration_weight': 1.0,     # Exploration weight for UCB formula
    'guided_strictness': 0.8,      # Strictness for guided rollouts (0.0-1.0)
    'max_time': 60,                # Maximum time in seconds
    
    # Minimax specific settings (only used if algorithm is 'minimax')
    'depth': 5,                    # Search depth for Minimax
    'randomize': True              # Whether to randomize equal moves
}

#-----------------------------------------------
# HELPER FUNCTIONS
#-----------------------------------------------

def format_board(board):
    """Format the board state for display."""
    symbols = {
        None: '.',
        'TIGER': 'T',
        'GOAT': 'G'
    }
    
    result = []
    for row in board:
        row_str = []
        for cell in row:
            if cell is None:
                row_str.append('.')
            else:
                row_str.append(symbols[cell["type"]])
        result.append(' '.join(row_str))
    
    return '\n'.join(result)

def parse_move_history(move_history, verbose=True):
    """Parse and explain each move from the move history."""
    moves = move_history.split(',')
    parsed_moves = []
    
    for i, move in enumerate(moves):
        turn = "GOAT" if i % 2 == 0 else "TIGER"
        move_desc = f"Move {i+1} ({turn}): "
        
        if move.startswith('p'):
            # Placement move
            x, y = int(move[1]), int(move[2])
            move_desc += f"Place goat at ({x}, {y})"
        elif move.startswith('m'):
            # Movement move
            from_x, from_y = int(move[1]), int(move[2])
            to_x, to_y = int(move[3]), int(move[4])
            move_desc += f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y})"
            
            if 'c' in move:
                # Capture move
                cap_x, cap_y = int(move[6]), int(move[7])
                move_desc += f", capturing goat at ({cap_x}, {cap_y})"
        
        parsed_moves.append(move_desc)
    
    if verbose:
        return '\n'.join(parsed_moves)
    else:
        return parsed_moves

def main():
    """Run a single test game."""
    print("=== Bagh Chal Simulation Test ===")
    print("\nTiger Agent:")
    pprint.pprint(TIGER_CONFIG)
    
    print("\nGoat Agent:")
    pprint.pprint(GOAT_CONFIG)
    
    print("\nStarting game simulation...")
    start_time = time.time()
    
    # Run the game
    runner = GameRunner(TIGER_CONFIG, GOAT_CONFIG)
    result = runner.run_game()
    
    total_time = time.time() - start_time
    print(f"\nSimulation completed in {total_time:.2f} seconds")
    
    print("\n=== Game Results ===")
    print(f"Winner: {result['winner']}")
    print(f"Reason: {result['reason']}")
    print(f"Total moves: {result['moves']}")
    print(f"Game duration: {result['game_duration']:.2f} seconds")
    print(f"Tiger average move time: {result['avg_tiger_move_time']:.4f} seconds")
    print(f"Goat average move time: {result['avg_goat_move_time']:.4f} seconds")
    print(f"Goats captured: {result['goats_captured']}")
    
    if result['first_capture_move'] is not None:
        print(f"First capture at move: {result['first_capture_move'] + 1}")
    else:
        print("No captures occurred")
        
    if result['phase_transition_move'] is not None:
        print(f"Phase transition at move: {result['phase_transition_move'] + 1}")
    
    print("\n=== Move History ===")
    print(parse_move_history(result['move_history']))
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 