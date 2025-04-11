import os
import time
import csv
import json
import itertools
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .game_runner import GameRunner

class SimulationController:
    """
    Manages running multiple games between agent configurations and saving results.
    """
    
    def __init__(self, output_dir: str = "simulation_results"):
        """
        Initialize the simulation controller.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different simulation types
        self.mcts_tournament_dir = os.path.join(output_dir, "mcts_tournament")
        self.main_competition_dir = os.path.join(output_dir, "main_competition")
        
        os.makedirs(self.mcts_tournament_dir, exist_ok=True)
        os.makedirs(self.main_competition_dir, exist_ok=True)
        
        # CSV headers
        self.csv_headers = [
            "game_id", "winner", "reason", "moves", "game_duration",
            "avg_tiger_move_time", "avg_goat_move_time", "first_capture_move",
            "goats_captured", "phase_transition_move", "move_history",
            "tiger_algorithm", "tiger_config", "goat_algorithm", "goat_config"
        ]
    
    def run_mcts_tournament(self, 
                         rollout_policies: List[str] = ["random", "lightweight", "guided"],
                         iterations: List[int] = [10000, 15000, 20000],
                         rollout_depths: List[int] = [4, 6],
                         games_per_matchup: int = 40,
                         start_idx: int = 0,
                         end_idx: int = None) -> str:
        """
        Run a tournament between all combinations of MCTS configurations.
        
        Args:
            rollout_policies: List of rollout policies to test
            iterations: List of iteration counts to test
            rollout_depths: List of rollout depths to test
            games_per_matchup: Number of games to play per matchup
            start_idx: Starting index of matchup to process (for parallelization)
            end_idx: Ending index of matchup to process (for parallelization)
            
        Returns:
            Path to the CSV file with results
        """
        # Generate all MCTS configurations
        mcts_configs = []
        for policy in rollout_policies:
            for iteration in iterations:
                for depth in rollout_depths:
                    config = {
                        'algorithm': 'mcts',
                        'rollout_policy': policy,
                        'iterations': iteration,
                        'rollout_depth': depth,
                        'exploration_weight': 1.0,  # Fixed for tournament
                        'guided_strictness': 0.8    # Fixed for tournament
                    }
                    mcts_configs.append(config)
        
        # Generate all matchups (each config plays against every other)
        all_matchups = list(itertools.combinations(mcts_configs, 2))
        
        # Allow processing a subset of matchups for parallel execution
        if end_idx is None:
            end_idx = len(all_matchups)
        
        matchups_to_process = all_matchups[start_idx:end_idx]
        print(f"Processing MCTS tournament matchups {start_idx} to {end_idx-1} " 
              f"({len(matchups_to_process)} of {len(all_matchups)} total matchups)")
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.mcts_tournament_dir, f"mcts_tournament_{timestamp}_{start_idx}_{end_idx}.csv")
        
        # Setup CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            writer.writeheader()
            
            # Process each matchup
            for i, (config1, config2) in enumerate(matchups_to_process):
                print(f"Matchup {start_idx + i + 1}/{end_idx}: "
                      f"{config1['rollout_policy']}-{config1['iterations']}-{config1['rollout_depth']} vs "
                      f"{config2['rollout_policy']}-{config2['iterations']}-{config2['rollout_depth']}")
                
                # Play games in both directions (each agent plays as Tiger and as Goat)
                games_played = 0
                half_games = games_per_matchup // 2
                
                # First half: config1 as Tiger, config2 as Goat
                for _ in range(half_games):
                    runner = GameRunner(config1, config2)
                    game_result = runner.run_game()
                    
                    # Prepare row data
                    row = {
                        "game_id": game_result["game_id"],
                        "winner": game_result["winner"],
                        "reason": game_result["reason"],
                        "moves": game_result["moves"],
                        "game_duration": game_result["game_duration"],
                        "avg_tiger_move_time": game_result["avg_tiger_move_time"],
                        "avg_goat_move_time": game_result["avg_goat_move_time"],
                        "first_capture_move": game_result["first_capture_move"],
                        "goats_captured": game_result["goats_captured"],
                        "phase_transition_move": game_result["phase_transition_move"],
                        "move_history": game_result["move_history"],
                        "tiger_algorithm": "mcts",
                        "tiger_config": json.dumps(config1),
                        "goat_algorithm": "mcts",
                        "goat_config": json.dumps(config2)
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    games_played += 1
                
                # Second half: config2 as Tiger, config1 as Goat
                for _ in range(half_games):
                    runner = GameRunner(config2, config1)
                    game_result = runner.run_game()
                    
                    # Prepare row data
                    row = {
                        "game_id": game_result["game_id"],
                        "winner": game_result["winner"],
                        "reason": game_result["reason"],
                        "moves": game_result["moves"],
                        "game_duration": game_result["game_duration"],
                        "avg_tiger_move_time": game_result["avg_tiger_move_time"],
                        "avg_goat_move_time": game_result["avg_goat_move_time"],
                        "first_capture_move": game_result["first_capture_move"],
                        "goats_captured": game_result["goats_captured"],
                        "phase_transition_move": game_result["phase_transition_move"],
                        "move_history": game_result["move_history"],
                        "tiger_algorithm": "mcts",
                        "tiger_config": json.dumps(config2),
                        "goat_algorithm": "mcts",
                        "goat_config": json.dumps(config1)
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    games_played += 1
                
                print(f"  Completed {games_played} games")
        
        print(f"MCTS tournament complete. Results saved to {output_file}")
        return output_file
    
    def run_main_competition(self, 
                          best_mcts_config: Dict,
                          minimax_depths: List[int] = [4, 5, 6],
                          games_per_matchup: int = 1000) -> str:
        """
        Run the main competition between the best MCTS configuration and Minimax at different depths.
        
        Args:
            best_mcts_config: The best MCTS configuration from the tournament
            minimax_depths: Depths to test for Minimax
            games_per_matchup: Number of games to play per matchup
            
        Returns:
            Path to the CSV file with results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.main_competition_dir, f"main_competition_{timestamp}.csv")
        
        # Setup CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            writer.writeheader()
            
            # Process each Minimax depth
            for depth in minimax_depths:
                minimax_config = {
                    'algorithm': 'minimax',
                    'depth': depth,
                    'randomize': True
                }
                
                print(f"Starting matchup: Minimax depth {depth} vs. Best MCTS")
                
                # Play games in both directions (each agent plays as Tiger and as Goat)
                half_games = games_per_matchup // 2
                games_played = 0
                
                # First half: Minimax as Tiger, MCTS as Goat
                for game_num in range(half_games):
                    if game_num % 10 == 0:
                        print(f"  Progress: {game_num}/{half_games} games (Minimax as Tiger)")
                    
                    runner = GameRunner(minimax_config, best_mcts_config)
                    game_result = runner.run_game()
                    
                    # Prepare row data
                    row = {
                        "game_id": game_result["game_id"],
                        "winner": game_result["winner"],
                        "reason": game_result["reason"],
                        "moves": game_result["moves"],
                        "game_duration": game_result["game_duration"],
                        "avg_tiger_move_time": game_result["avg_tiger_move_time"],
                        "avg_goat_move_time": game_result["avg_goat_move_time"],
                        "first_capture_move": game_result["first_capture_move"],
                        "goats_captured": game_result["goats_captured"],
                        "phase_transition_move": game_result["phase_transition_move"],
                        "move_history": game_result["move_history"],
                        "tiger_algorithm": "minimax",
                        "tiger_config": json.dumps(minimax_config),
                        "goat_algorithm": "mcts",
                        "goat_config": json.dumps(best_mcts_config)
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    games_played += 1
                
                # Second half: MCTS as Tiger, Minimax as Goat
                for game_num in range(half_games):
                    if game_num % 10 == 0:
                        print(f"  Progress: {game_num}/{half_games} games (MCTS as Tiger)")
                    
                    runner = GameRunner(best_mcts_config, minimax_config)
                    game_result = runner.run_game()
                    
                    # Prepare row data
                    row = {
                        "game_id": game_result["game_id"],
                        "winner": game_result["winner"],
                        "reason": game_result["reason"],
                        "moves": game_result["moves"],
                        "game_duration": game_result["game_duration"],
                        "avg_tiger_move_time": game_result["avg_tiger_move_time"],
                        "avg_goat_move_time": game_result["avg_goat_move_time"],
                        "first_capture_move": game_result["first_capture_move"],
                        "goats_captured": game_result["goats_captured"],
                        "phase_transition_move": game_result["phase_transition_move"],
                        "move_history": game_result["move_history"],
                        "tiger_algorithm": "mcts",
                        "tiger_config": json.dumps(best_mcts_config),
                        "goat_algorithm": "minimax",
                        "goat_config": json.dumps(minimax_config)
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    games_played += 1
                
                print(f"  Completed {games_played} games for Minimax depth {depth}")
        
        print(f"Main competition complete. Results saved to {output_file}")
        return output_file 