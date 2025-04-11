import os
import time
import csv
import json
import itertools
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import glob

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
    
    def _find_existing_results(self, output_path: str) -> Set[str]:
        """
        Find existing game IDs from a CSV file to avoid duplication.
        
        Args:
            output_path: Path to the CSV file
            
        Returns:
            Set of existing game IDs
        """
        existing_game_ids = set()
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if 'game_id' in row:
                            existing_game_ids.add(row['game_id'])
            except Exception as e:
                print(f"Warning: Error reading existing file {output_path}: {e}")
                print("Starting fresh (existing results will be preserved)")
        
        return existing_game_ids
    
    def _find_existing_matchups(self, output_path: str) -> Set[Tuple[str, str]]:
        """
        Find existing matchup signatures from a CSV file to track completion.
        A matchup signature is a tuple of (tiger_config_str, goat_config_str).
        
        Args:
            output_path: Path to the CSV file
            
        Returns:
            Set of completed matchup signatures with game counts
        """
        # Track matchups and the number of games played for each
        matchup_counts = {}
        
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if 'tiger_config' in row and 'goat_config' in row:
                            matchup_key = (row['tiger_config'], row['goat_config'])
                            matchup_counts[matchup_key] = matchup_counts.get(matchup_key, 0) + 1
            except Exception as e:
                print(f"Warning: Error reading existing file {output_path}: {e}")
                print("Starting fresh (existing results will be preserved)")
        
        return matchup_counts
    
    def run_mcts_tournament(self, 
                          rollout_policies: List[str] = ["random", "lightweight", "guided"],
                          iterations: List[int] = [10000, 15000, 20000],
                          rollout_depths: List[int] = [4, 6],
                          games_per_matchup: int = 40,
                          start_idx: int = 0,
                          end_idx: int = None,
                          output_file: str = None) -> str:
        """
        Run a tournament between all combinations of MCTS configurations.
        
        Args:
            rollout_policies: List of rollout policies to test
            iterations: List of iteration counts to test
            rollout_depths: List of rollout depths to test
            games_per_matchup: Number of games to play per matchup
            start_idx: Starting index of matchup to process (for parallelization)
            end_idx: Ending index of matchup to process (for parallelization)
            output_file: Optional specific output file path (for resuming)
            
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
        
        # Create or find output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.mcts_tournament_dir, f"mcts_tournament_{timestamp}_{start_idx}_{end_idx}.csv")
        else:
            # If specific output file provided, use it for resumption
            output_file = os.path.join(self.mcts_tournament_dir, output_file)
        
        # Get existing game IDs to avoid duplication
        existing_game_ids = self._find_existing_results(output_file)
        
        # Get existing matchup counts to know how many games are already done
        matchup_counts = self._find_existing_matchups(output_file)
        
        print(f"Found {len(existing_game_ids)} existing games in output file")
        
        # Setup CSV file - if it exists, open in append mode
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            
            # Only write header if creating a new file
            if not file_exists:
                writer.writeheader()
            
            # Process each matchup
            for i, (config1, config2) in enumerate(matchups_to_process):
                config1_str = json.dumps(config1)
                config2_str = json.dumps(config2)
                
                # Calculate how many games we've already played for this matchup in each direction
                tiger1_goat2_played = matchup_counts.get((config1_str, config2_str), 0)
                tiger2_goat1_played = matchup_counts.get((config2_str, config1_str), 0)
                
                half_games = games_per_matchup // 2
                
                print(f"Matchup {start_idx + i + 1}/{end_idx}: "
                      f"{config1['rollout_policy']}-{config1['iterations']}-{config1['rollout_depth']} vs "
                      f"{config2['rollout_policy']}-{config2['iterations']}-{config2['rollout_depth']}")
                print(f"  Already played: {tiger1_goat2_played}/{half_games} (config1 as Tiger) and "
                      f"{tiger2_goat1_played}/{half_games} (config2 as Tiger)")
                
                # Track games completed in this session
                games_played_this_session = 0
                
                # First half: config1 as Tiger, config2 as Goat
                for _ in range(half_games - tiger1_goat2_played):
                    runner = GameRunner(config1, config2)
                    game_result = runner.run_game()
                    
                    # Skip if we've already recorded this game
                    if game_result["game_id"] in existing_game_ids:
                        continue
                    
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
                        "tiger_config": config1_str,
                        "goat_algorithm": "mcts",
                        "goat_config": config2_str
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    # Track this game as completed
                    existing_game_ids.add(game_result["game_id"])
                    games_played_this_session += 1
                
                # Second half: config2 as Tiger, config1 as Goat
                for _ in range(half_games - tiger2_goat1_played):
                    runner = GameRunner(config2, config1)
                    game_result = runner.run_game()
                    
                    # Skip if we've already recorded this game
                    if game_result["game_id"] in existing_game_ids:
                        continue
                    
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
                        "tiger_config": config2_str,
                        "goat_algorithm": "mcts",
                        "goat_config": config1_str
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    # Track this game as completed
                    existing_game_ids.add(game_result["game_id"])
                    games_played_this_session += 1
                
                print(f"  Completed {games_played_this_session} new games in this session")
        
        print(f"MCTS tournament progress saved to {output_file}")
        return output_file
    
    def run_main_competition(self, 
                          best_mcts_config: Dict,
                          minimax_depths: List[int] = [4, 5, 6],
                          games_per_matchup: int = 1000,
                          output_file: str = None) -> str:
        """
        Run the main competition between the best MCTS configuration and Minimax at different depths.
        
        Args:
            best_mcts_config: The best MCTS configuration from the tournament
            minimax_depths: Depths to test for Minimax
            games_per_matchup: Number of games to play per matchup
            output_file: Optional specific output file path (for resuming)
            
        Returns:
            Path to the CSV file with results
        """
        # Create or find output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.main_competition_dir, f"main_competition_{timestamp}.csv")
        else:
            # If specific output file provided, use it for resumption
            output_file = os.path.join(self.main_competition_dir, output_file)
        
        # Get existing game IDs to avoid duplication
        existing_game_ids = self._find_existing_results(output_file)
        
        # Get existing matchup counts to know how many games are already done
        matchup_counts = self._find_existing_matchups(output_file)
        
        print(f"Found {len(existing_game_ids)} existing games in output file")
        
        # Setup CSV file - if it exists, open in append mode
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            
            # Only write header if creating a new file
            if not file_exists:
                writer.writeheader()
            
            # Convert MCTS config to string for matching
            mcts_config_str = json.dumps(best_mcts_config)
            
            # Process each Minimax depth
            for depth in minimax_depths:
                minimax_config = {
                    'algorithm': 'minimax',
                    'depth': depth,
                    'randomize': True
                }
                
                minimax_config_str = json.dumps(minimax_config)
                
                # Calculate how many games we've already played for this matchup in each direction
                minimax_tiger_played = matchup_counts.get((minimax_config_str, mcts_config_str), 0)
                mcts_tiger_played = matchup_counts.get((mcts_config_str, minimax_config_str), 0)
                
                half_games = games_per_matchup // 2
                
                print(f"Matchup: Minimax depth {depth} vs. Best MCTS")
                print(f"  Already played: {minimax_tiger_played}/{half_games} (Minimax as Tiger) and "
                      f"{mcts_tiger_played}/{half_games} (MCTS as Tiger)")
                
                # Track games completed in this session
                games_played_this_session = 0
                
                # First half: Minimax as Tiger, MCTS as Goat
                for game_num in range(half_games - minimax_tiger_played):
                    if game_num % 10 == 0:
                        print(f"  Progress: {game_num}/{half_games - minimax_tiger_played} games (Minimax as Tiger)")
                    
                    runner = GameRunner(minimax_config, best_mcts_config)
                    game_result = runner.run_game()
                    
                    # Skip if we've already recorded this game
                    if game_result["game_id"] in existing_game_ids:
                        continue
                    
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
                        "tiger_config": minimax_config_str,
                        "goat_algorithm": "mcts",
                        "goat_config": mcts_config_str
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    # Track this game as completed
                    existing_game_ids.add(game_result["game_id"])
                    games_played_this_session += 1
                
                # Second half: MCTS as Tiger, Minimax as Goat
                for game_num in range(half_games - mcts_tiger_played):
                    if game_num % 10 == 0:
                        print(f"  Progress: {game_num}/{half_games - mcts_tiger_played} games (MCTS as Tiger)")
                    
                    runner = GameRunner(best_mcts_config, minimax_config)
                    game_result = runner.run_game()
                    
                    # Skip if we've already recorded this game
                    if game_result["game_id"] in existing_game_ids:
                        continue
                    
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
                        "tiger_config": mcts_config_str,
                        "goat_algorithm": "minimax",
                        "goat_config": minimax_config_str
                    }
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    # Track this game as completed
                    existing_game_ids.add(game_result["game_id"])
                    games_played_this_session += 1
                
                print(f"  Completed {games_played_this_session} new games for Minimax depth {depth}")
        
        print(f"Main competition progress saved to {output_file}")
        return output_file
        
    def find_most_recent_tournament_file(self, start_idx, end_idx):
        """
        Find the most recent tournament file for a given index range.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            
        Returns:
            The filename of the most recent tournament file or None if not found
        """
        pattern = os.path.join(self.mcts_tournament_dir, f"mcts_tournament_*_{start_idx}_{end_idx}.csv")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None
            
        # Get the most recent file by modification time
        return os.path.basename(max(matching_files, key=os.path.getmtime)) 