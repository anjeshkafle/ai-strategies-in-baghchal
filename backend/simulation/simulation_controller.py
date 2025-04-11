import os
import time
import csv
import json
import itertools
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import glob
import multiprocessing as mp

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
    
    def _run_game_wrapper(self, args):
        """
        Wrapper function to run a single game for multiprocessing.
        
        Args:
            args: Tuple containing (config1, config2, is_config1_tiger, config1_str, config2_str)
            
        Returns:
            Game result dictionary with added metadata about tiger/goat roles
        """
        config1, config2, is_config1_tiger, config1_str, config2_str = args
        
        if is_config1_tiger:
            tiger_config = config1
            goat_config = config2
            tiger_config_str = config1_str
            goat_config_str = config2_str
        else:
            tiger_config = config2
            goat_config = config1
            tiger_config_str = config2_str
            goat_config_str = config1_str
        
        runner = GameRunner(tiger_config, goat_config)
        result = runner.run_game()
        
        # Add role information to the result for proper tracking
        result["tiger_config_str"] = tiger_config_str
        result["goat_config_str"] = goat_config_str
        result["is_config1_tiger"] = is_config1_tiger
        
        return result
    
    def find_most_recent_tournament_file(self, start_idx=None, end_idx=None):
        """
        Find the most recent tournament file for a given index range.
        If no indices are provided, finds the most recent file overall.
        
        Args:
            start_idx: Optional starting index
            end_idx: Optional ending index
            
        Returns:
            The full path of the most recent tournament file or None if not found
        """
        if start_idx is not None and end_idx is not None:
            # Look for an exact match first
            pattern = os.path.join(self.mcts_tournament_dir, f"mcts_tournament_*_{start_idx}_{end_idx}.csv")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # Get the most recent file by modification time
                most_recent = max(matching_files, key=os.path.getmtime)
                print(f"Found existing tournament file for range {start_idx}-{end_idx}: {os.path.basename(most_recent)}")
                return most_recent
            
        # If no exact match is found or no indices provided, find the most recent file
        pattern = os.path.join(self.mcts_tournament_dir, "mcts_tournament_*.csv")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print("No existing tournament files found")
            return None
        
        # Get the most recent file by modification time
        most_recent = max(matching_files, key=os.path.getmtime)
        print(f"Found most recent tournament file: {os.path.basename(most_recent)}")
        return most_recent
    
    def run_mcts_tournament(self, 
                          rollout_policies: List[str] = ["random", "lightweight", "guided"],
                          iterations: List[int] = [10000, 15000, 20000],
                          rollout_depths: List[int] = [4, 6],
                          games_per_matchup: int = 40,
                          start_idx: int = 0,
                          end_idx: int = None,
                          output_file: str = None,
                          parallel_games: int = None) -> str:
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
            parallel_games: Number of games to run in parallel (defaults to CPU count)
            
        Returns:
            Path to the CSV file with results
        """
        # Set default parallel games to CPU count - 1 (leave one core free)
        if parallel_games is None:
            parallel_games = max(1, mp.cpu_count() - 1)
        
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
                        'exploration_weight': 1.0,
                        'guided_strictness': 0.8
                    }
                    mcts_configs.append(config)
        
        # Generate all matchups (each config plays against every other)
        all_matchups = list(itertools.combinations(mcts_configs, 2))
        
        # Allow processing a subset of matchups for parallel execution
        if end_idx is None:
            end_idx = len(all_matchups)
        
        matchups_to_process = all_matchups[start_idx:end_idx]
        total_matchups = len(matchups_to_process)
        total_games = total_matchups * games_per_matchup
        
        print(f"\nMCTS Tournament Setup:")
        print(f"  Total configurations: {len(mcts_configs)}")
        print(f"  Total matchups to process: {total_matchups}")
        print(f"  Games per matchup: {games_per_matchup}")
        print(f"  Total games to play: {total_games}")
        print(f"  Processing matchups {start_idx} to {end_idx-1}")
        print(f"  Running {parallel_games} games in parallel")
        print()
        
        # Create or find output file
        if output_file is None:
            # Try to find an existing file for this range to resume from
            existing_file = self.find_most_recent_tournament_file(start_idx, end_idx)
            if existing_file:
                output_file = existing_file
                print(f"Resuming from existing file: {os.path.basename(output_file)}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.mcts_tournament_dir, f"mcts_tournament_{timestamp}_{start_idx}_{end_idx}.csv")
                print(f"Creating new output file: {os.path.basename(output_file)}")
        else:
            # If specific output file provided, check if it exists
            full_path = os.path.join(self.mcts_tournament_dir, output_file)
            if os.path.exists(full_path):
                output_file = full_path
                print(f"Resuming from specified file: {os.path.basename(output_file)}")
            else:
                output_file = os.path.join(self.mcts_tournament_dir, output_file)
                print(f"Creating specified output file: {os.path.basename(output_file)}")
        
        # Get existing game IDs to avoid duplication
        existing_game_ids = self._find_existing_results(output_file)
        existing_matchups = self._find_existing_matchups(output_file)
        
        print(f"Found {len(existing_game_ids)} existing games in output file")
        
        # For each matchup, calculate how many games we've already played
        total_existing_games = sum(existing_matchups.values())
        total_remaining_games = total_games - total_existing_games
        print(f"Total existing games: {total_existing_games}, Remaining games: {total_remaining_games}")
        
        # Setup CSV file
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a' if file_exists else 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            
            if not file_exists:
                writer.writeheader()
            
            # Track progress
            games_played = 0
            games_saved = 0
            last_progress_update = time.time()
            
            # Initialize process pool
            with mp.Pool(processes=parallel_games) as pool:
                # Process each matchup
                for i, (config1, config2) in enumerate(matchups_to_process):
                    config1_str = json.dumps(config1)
                    config2_str = json.dumps(config2)
                    
                    # Calculate how many games we've already played for this matchup
                    tiger1_goat2_played = existing_matchups.get((config1_str, config2_str), 0)
                    tiger2_goat1_played = existing_matchups.get((config2_str, config1_str), 0)
                    
                    half_games = games_per_matchup // 2
                    
                    print(f"\nMatchup {i+1}/{total_matchups}:")
                    print(f"  Config 1: {config1['rollout_policy']}-{config1['iterations']}-{config1['rollout_depth']}")
                    print(f"  Config 2: {config2['rollout_policy']}-{config2['iterations']}-{config2['rollout_depth']}")
                    print(f"  Already played: {tiger1_goat2_played}/{half_games} (config1 as Tiger) and "
                          f"{tiger2_goat1_played}/{half_games} (config2 as Tiger)")
                    
                    # Create tasks for first half: config1 as Tiger, config2 as Goat
                    tasks_config1_tiger = [(config1, config2, True, config1_str, config2_str) 
                                        for _ in range(half_games - tiger1_goat2_played)]
                    
                    # Create tasks for second half: config2 as Tiger, config1 as Goat
                    tasks_config2_tiger = [(config1, config2, False, config1_str, config2_str) 
                                        for _ in range(half_games - tiger2_goat1_played)]
                    
                    # Combine all tasks
                    all_tasks = tasks_config1_tiger + tasks_config2_tiger
                    
                    if not all_tasks:
                        print("  All games for this matchup already completed, skipping")
                        continue
                    
                    # Run all games in parallel
                    batch_start_time = time.time()
                    print(f"  Starting {len(all_tasks)} games in parallel...")
                    print(f"  Config1 as Tiger: {len(tasks_config1_tiger)} games, Config2 as Tiger: {len(tasks_config2_tiger)} games")
                    
                    for game_result in pool.imap_unordered(self._run_game_wrapper, all_tasks):
                        games_played += 1
                        
                        if game_result["game_id"] in existing_game_ids:
                            continue
                        
                        # Use the role information that was added in the wrapper
                        tiger_config_str = game_result["tiger_config_str"]
                        goat_config_str = game_result["goat_config_str"]
                        
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
                            "tiger_config": tiger_config_str,
                            "goat_algorithm": "mcts",
                            "goat_config": goat_config_str
                        }
                        
                        writer.writerow(row)
                        csvfile.flush()
                        games_saved += 1
                        existing_game_ids.add(game_result["game_id"])
                        
                        # Update progress occasionally
                        if time.time() - last_progress_update > 5:
                            config1_tiger_count = sum(1 for t in all_tasks if t[2] is True and t in tasks_config1_tiger)
                            config2_tiger_count = sum(1 for t in all_tasks if t[2] is False and t in tasks_config2_tiger)
                            print(f"  Progress: {games_played}/{len(all_tasks)} games completed, {games_saved} saved")
                            print(f"  Remaining: Config1 as Tiger: {config1_tiger_count}, Config2 as Tiger: {config2_tiger_count}")
                            last_progress_update = time.time()
                    
                    # Count how many of each type we ended up with
                    config1_tiger_played = existing_matchups.get((config1_str, config2_str), 0) + len(tasks_config1_tiger) - tiger1_goat2_played
                    config2_tiger_played = existing_matchups.get((config2_str, config1_str), 0) + len(tasks_config2_tiger) - tiger2_goat1_played
                    
                    batch_time = time.time() - batch_start_time
                    print(f"  Completed {len(all_tasks)} games in {batch_time:.2f} seconds ({len(all_tasks)/batch_time:.2f} games/sec)")
                    print(f"  Current totals: Config1 as Tiger: {config1_tiger_played}/{half_games}, Config2 as Tiger: {config2_tiger_played}/{half_games}")
        
        print(f"\nTournament complete!")
        print(f"Total games played: {games_played}")
        print(f"Total games saved: {games_saved}")
        print(f"Results saved to: {output_file}")
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