#!/usr/bin/env python3
"""
Interactive game player for recorded games.
Loads games from MCTS tournament or main competition results and allows step-by-step playback.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Optional
import readchar  # For keyboard input without requiring Enter

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.game_state import GameState
from models.mcts_agent import MCTSAgent
from models.minimax_agent import MinimaxAgent

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def create_initial_board():
    """Create a board with tigers in initial positions."""
    board = [[None for _ in range(5)] for _ in range(5)]
    # Place tigers in corners
    board[0][0] = {"type": "TIGER"}
    board[0][4] = {"type": "TIGER"}
    board[4][0] = {"type": "TIGER"}
    board[4][4] = {"type": "TIGER"}
    return board

def display_board_and_info(game_state: GameState, move_number: int, total_moves: int, game_data: Dict):
    """Display the board and game information side by side."""
    board = game_state.board
    
    # Header
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Move {move_number}/{total_moves} - Bagh-Chal Game Replay{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    
    # Board display with info on the right
    print("Current Board State:" + " " * 11 + f"{BOLD}Game State:{RESET}" + " " * 25 + f"{BOLD}Move Info:{RESET}")
    print("-" * 11 + " " * 20 + f"Phase: {game_state.phase}" + " " * 20 + f"Turn: {game_state.turn}")
    
    for y in range(5):
        # Board row
        row = "|"
        for x in range(5):
            piece = board[y][x]
            if piece is None:
                row += " |"
            elif piece["type"] == "TIGER":
                row += f"{RED}T{RESET}|"
            else:  # GOAT
                row += f"{GREEN}G{RESET}|"
        
        # Add state info on the right
        if y == 0:
            row += " " * 20 + f"Goats placed: {game_state.goats_placed}/20" + " " * 18 + f"Goats captured: {game_state.goats_captured}"
        elif y == 1:
            row += " " * 20 + f"Winner: {game_data['winner']}" + " " * 24 + f"Reason: {game_data['reason']}"
        elif y == 2 and move_number > 0 and move_number <= len(game_data['move_history']):
            move = game_data['move_history'][move_number-1]
            move_str = f"{RED if move['player'] == 'TIGER' else GREEN}{move['player']}{RESET} "
            if move['type'] == 'movement':
                move_str += f"({move['from']['x']},{move['from']['y']})→({move['to']['x']},{move['to']['y']})"
            else:
                move_str += f"place({move['x']},{move['y']})"
            row += " " * 20 + move_str
        elif y == 3:
            row += " " * 20 + f"Controls: → next, ← prev, q quit"
            
        print(row)
        print("-" * 11)

def find_game(game_id: str) -> Optional[str]:
    """Find a game by ID in either tournament or competition results."""
    # Check MCTS tournament results
    mcts_dir = os.path.join("simulation_results", "mcts_tournament")
    if os.path.exists(mcts_dir):
        for file in os.listdir(mcts_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(mcts_dir, file))
                if game_id in df['game_id'].values:
                    return os.path.join(mcts_dir, file)
    
    # Check main competition results
    main_dir = os.path.join("simulation_results", "main_competition")
    if os.path.exists(main_dir):
        for file in os.listdir(main_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(main_dir, file))
                if game_id in df['game_id'].values:
                    return os.path.join(main_dir, file)
    
    return None

def parse_move_history(move_str: str) -> List[Dict]:
    """Parse the compact move history string into move objects."""
    moves = []
    current_player = "GOAT"  # Goats move first in placement phase
    for move in move_str.split(','):
        move_type = move[0]
        if move_type == 'p':  # placement
            x, y = int(move[1]), int(move[2])
            moves.append({
                'type': 'placement',
                'player': current_player,
                'x': x,
                'y': y
            })
        elif move_type == 'm':  # movement
            from_x = int(move[1])
            from_y = int(move[2])
            to_x = int(move[3])
            to_y = int(move[4])
            moves.append({
                'type': 'movement',
                'player': current_player,
                'from': {'x': from_x, 'y': from_y},
                'to': {'x': to_x, 'y': to_y}
            })
        # Switch player for next move
        current_player = "GOAT" if current_player == "TIGER" else "TIGER"
    return moves

def load_game(game_id: str) -> Dict:
    """Load a game by ID from the results files."""
    file_path = find_game(game_id)
    if not file_path:
        raise ValueError(f"Game {game_id} not found in any results files")
    
    df = pd.read_csv(file_path)
    game_row = df[df['game_id'] == game_id].iloc[0]
    
    # Parse move history from compact format
    move_history = parse_move_history(game_row['move_history'])
    
    # Convert string representations to Python objects
    game_data = {
        'game_id': game_row['game_id'],
        'winner': game_row['winner'],
        'reason': game_row['reason'],
        'moves': game_row['moves'],
        'game_duration': game_row['game_duration'],
        'avg_tiger_move_time': game_row['avg_tiger_move_time'],
        'avg_goat_move_time': game_row['avg_goat_move_time'],
        'first_capture_move': game_row['first_capture_move'],
        'goats_captured': game_row['goats_captured'],
        'phase_transition_move': game_row['phase_transition_move'],
        'move_history': move_history,
        'tiger_algorithm': game_row['tiger_algorithm'],
        'tiger_config': json.loads(game_row['tiger_config']),
        'goat_algorithm': game_row['goat_algorithm'],
        'goat_config': json.loads(game_row['goat_config'])
    }
    
    return game_data

def play_game(game_id: str):
    """Interactive game playback."""
    try:
        game_data = load_game(game_id)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    current_move = 0
    total_moves = len(game_data['move_history'])
    game_state = GameState()
    game_state.board = create_initial_board()  # Set initial tiger positions
    game_state.phase = "PLACEMENT"
    game_state.turn = "GOAT"
    game_state.goats_placed = 0
    game_state.goats_captured = 0
    
    while True:
        clear_screen()
        display_board_and_info(game_state, current_move, total_moves, game_data)
        
        key = readchar.readkey()
        
        if key == 'q':
            break
        elif key == '\x1b[C':  # Right arrow
            if current_move < total_moves:
                move = game_data['move_history'][current_move]
                game_state.apply_move(move)
                current_move += 1
        elif key == '\x1b[D':  # Left arrow
            if current_move > 0:
                current_move -= 1
                # Reset and replay up to current move
                game_state = GameState()
                game_state.board = create_initial_board()  # Reset with initial tigers
                game_state.phase = "PLACEMENT"
                game_state.turn = "GOAT"
                game_state.goats_placed = 0
                game_state.goats_captured = 0
                for i in range(current_move):
                    move = game_data['move_history'][i]
                    game_state.apply_move(move)

def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python -m tests.game_player <game_id>")
        return
    
    game_id = sys.argv[1]
    play_game(game_id)

if __name__ == "__main__":
    main() 