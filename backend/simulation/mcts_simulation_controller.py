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
from .google_sheets_sync import GoogleSheetsSync

class MCTSSimulationController:
    """
    Manages running multiple games between MCTS agent configurations and saving results.
    """
    
    def __init__(self, output_dir: str = "simulation_results", google_sheets_url: str = None, batch_size: int = 100):
        """
        Initialize the simulation controller.
        
        Args:
            output_dir: Directory to save results
            google_sheets_url: URL for Google Sheets sync, or None to disable
            batch_size: Batch size for Google Sheets sync
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different simulation types
        self.mcts_tournament_dir = os.path.join(output_dir, "mcts_tournament")
        
        os.makedirs(self.mcts_tournament_dir, exist_ok=True)
        
        # CSV headers
        self.csv_headers = [
            "game_id", "winner", "reason", "moves", "game_duration",
            "avg_tiger_move_time", "avg_goat_move_time", "first_capture_move",
            "goats_captured", "phase_transition_move", "move_history",
            "tiger_algorithm", "tiger_config", "goat_algorithm", "goat_config"
        ]
        
        # Initialize Google Sheets sync
        self.sheets_sync = GoogleSheetsSync(
            webapp_url=google_sheets_url, 
            batch_size=batch_size,
            output_dir=output_dir
        )
    
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
                          mcts_configs: List[Dict] = None,
                          max_simulation_time: int = 60,  # Maximum time in minutes
                          start_idx: int = 0,
                          end_idx: int = None,
                          output_file: str = None,
                          parallel_games: int = None) -> str:
        """
        Run a tournament between all combinations of MCTS configurations.
        
        Args:
            mcts_configs: List of MCTS configurations to test
            max_simulation_time: Maximum time to run the simulation in minutes
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
        
        # Check that configurations are provided
        if mcts_configs is None:
            raise ValueError("mcts_configs must be provided")
        
        # Generate all matchups (each config plays against every other)
        all_matchups = list(itertools.combinations(mcts_configs, 2))
        
        # Allow processing a subset of matchups for parallel execution
        if end_idx is None:
            end_idx = len(all_matchups)
        
        matchups_to_process = all_matchups[start_idx:end_idx]
        total_matchups = len(matchups_to_process)
        
        print(f"\nMCTS Tournament Setup:")
        print(f"  Total configurations: {len(mcts_configs)}")
        print(f"  Total matchups to process: {total_matchups}")
        print(f"  Max simulation time: {max_simulation_time} minutes")
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
            # If specific output file provided, check if it's already a full path
            if os.path.isabs(output_file) or os.path.exists(output_file):
                # Use output_file as is if it's an absolute path or exists
                print(f"Using specified file: {os.path.basename(output_file)}")
            else:
                # Otherwise, join with tournament directory
                output_file = os.path.join(self.mcts_tournament_dir, output_file)
                print(f"Creating specified output file: {os.path.basename(output_file)}")
        
        # Get existing game IDs to avoid duplication
        existing_game_ids = self._find_existing_results(output_file)
        existing_matchups = self._find_existing_matchups(output_file)
        
        print(f"Found {len(existing_game_ids)} existing games in output file")
        
        # Calculate how many games exist per matchup for balanced scheduling
        matchup_game_counts = {}
        for i, (config1, config2) in enumerate(matchups_to_process):
            config1_str = json.dumps(config1)
            config2_str = json.dumps(config2)
            
            # Count games in both directions (config1 as tiger, config2 as tiger)
            tiger1_goat2_played = existing_matchups.get((config1_str, config2_str), 0)
            tiger2_goat1_played = existing_matchups.get((config2_str, config1_str), 0)
            
            # Store counts for each matchup
            matchup_idx = i
            matchup_game_counts[matchup_idx] = {
                'total': tiger1_goat2_played + tiger2_goat1_played,
                'tiger1_goat2': tiger1_goat2_played,
                'tiger2_goat1': tiger2_goat1_played,
                'tiger1_goat2_in_progress': 0,  # Track in-progress games
                'tiger2_goat1_in_progress': 0,  # Track in-progress games
                'config1': config1,
                'config2': config2,
                'config1_str': config1_str,
                'config2_str': config2_str
            }
        
        # Calculate the minimum and maximum games per matchup
        min_games = min(data['total'] for data in matchup_game_counts.values()) if matchup_game_counts else 0
        max_games = max(data['total'] for data in matchup_game_counts.values()) if matchup_game_counts else 0
        
        print(f"Existing games per matchup: min={min_games}, max={max_games}")
        
        # Calculate end time for the simulation
        end_time = time.time() + (max_simulation_time * 60)  # Convert minutes to seconds
        
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
            simulation_start_time = time.time()
            
            # Initialize thread lock
            import threading
            lock = threading.Lock()
            active_tasks = 0
            
            # Define helper functions outside the pool to avoid closure issues
            def get_next_tasks(num_tasks=1, pool_running=True):
                """Get the next highest priority tasks to execute."""
                with lock:
                    # Skip if we're out of time
                    if time.time() >= end_time:
                        return []
                    
                    # Calculate effective game counts (completed + in-progress)
                    for idx in matchup_game_counts:
                        matchup_game_counts[idx]['effective_total'] = (
                            matchup_game_counts[idx]['total'] + 
                            matchup_game_counts[idx]['tiger1_goat2_in_progress'] + 
                            matchup_game_counts[idx]['tiger2_goat1_in_progress']
                        )
                        matchup_game_counts[idx]['effective_tiger1'] = (
                            matchup_game_counts[idx]['tiger1_goat2'] + 
                            matchup_game_counts[idx]['tiger1_goat2_in_progress']
                        )
                        matchup_game_counts[idx]['effective_tiger2'] = (
                            matchup_game_counts[idx]['tiger2_goat1'] + 
                            matchup_game_counts[idx]['tiger2_goat1_in_progress']
                        )
                    
                    # Find the minimum effective total across all matchups
                    min_effective_total = min(data['effective_total'] for data in matchup_game_counts.values()) if matchup_game_counts else 0
                    
                    # Sort matchups by priority
                    # First prioritize matchups with fewer effective total games (but not more than 2 from min)
                    # Then prioritize by whichever side has fewer effective games 
                    matchup_priorities = sorted(
                        matchup_game_counts.items(),
                        key=lambda x: (
                            x[1]['effective_total'] > min_effective_total + 2,  # First prioritize games within 2 of min
                            x[1]['effective_total'],  # Then by effective total
                            min(x[1]['effective_tiger1'], x[1]['effective_tiger2'])  # Then by side with fewer games
                        )
                    )
                    
                    # Collect tasks to schedule
                    tasks = []
                    
                    # Process matchups until we have enough tasks
                    for matchup_idx, matchup_data in matchup_priorities:
                        if len(tasks) >= num_tasks:
                            break
                        
                        # Skip this matchup if it already has 2+ more games than minimum
                        if matchup_data['effective_total'] > min_effective_total + 2:
                            continue
                            
                        config1 = matchup_data['config1']
                        config2 = matchup_data['config2']
                        config1_str = matchup_data['config1_str']
                        config2_str = matchup_data['config2_str']
                        
                        # Determine which sides need games based on effective counts
                        tiger1_games = matchup_data['effective_tiger1']
                        tiger2_games = matchup_data['effective_tiger2']
                        
                        # Add tasks based on which side has fewer games
                        if tiger1_games <= tiger2_games and len(tasks) < num_tasks:
                            # Schedule config1 as tiger
                            tasks.append((config1, config2, True, config1_str, config2_str))
                            # Mark this task as in progress for future scheduling decisions
                            matchup_game_counts[matchup_idx]['tiger1_goat2_in_progress'] += 1
                            
                        elif tiger2_games <= tiger1_games and len(tasks) < num_tasks:
                            # Schedule config2 as tiger
                            tasks.append((config1, config2, False, config1_str, config2_str))
                            # Mark this task as in progress for future scheduling decisions
                            matchup_game_counts[matchup_idx]['tiger2_goat1_in_progress'] += 1
                    
                    # If we couldn't find tasks within the 2-game threshold but there are matchups,
                    # allow scheduling of additional tasks from the lowest-count matchups
                    if not tasks and matchup_priorities:
                        # Find matchups with the lowest effective total
                        lowest_total = matchup_priorities[0][1]['effective_total']
                        lowest_matchups = [m for m in matchup_priorities if m[1]['effective_total'] == lowest_total]
                        
                        # Schedule from these matchups
                        for matchup_idx, matchup_data in lowest_matchups:
                            if len(tasks) >= num_tasks:
                                break
                                
                            config1 = matchup_data['config1']
                            config2 = matchup_data['config2']
                            config1_str = matchup_data['config1_str']
                            config2_str = matchup_data['config2_str']
                            
                            # Determine which sides need games based on effective counts
                            tiger1_games = matchup_data['effective_tiger1']
                            tiger2_games = matchup_data['effective_tiger2']
                            
                            # Balance sides
                            if tiger1_games <= tiger2_games and len(tasks) < num_tasks:
                                tasks.append((config1, config2, True, config1_str, config2_str))
                                matchup_game_counts[matchup_idx]['tiger1_goat2_in_progress'] += 1
                                
                            elif tiger2_games <= tiger1_games and len(tasks) < num_tasks:
                                tasks.append((config1, config2, False, config1_str, config2_str))
                                matchup_game_counts[matchup_idx]['tiger2_goat1_in_progress'] += 1
                    
                    return tasks
            
            def get_config_summary(config):
                """Generate a short readable summary of an agent configuration"""
                if 'rollout_policy' not in config:
                    return "Unknown config"
                
                # Extract key parameters
                policy = config.get('rollout_policy', 'unknown')
                depth = config.get('rollout_depth', '?')
                exploration = config.get('exploration_weight', 1.414)
                
                # Create a readable string
                summary = f"{policy.capitalize()}"
                if policy == "guided":
                    strictness = config.get('guided_strictness', 0.5)
                    summary += f"-{strictness}"
                summary += f" d{depth} e{exploration}"
                
                return summary
                
            def on_game_complete(game_result):
                """Callback when a game completes - process results and schedule next task."""
                nonlocal games_played, games_saved, active_tasks, last_progress_update
                
                with lock:
                    active_tasks -= 1
                    games_played += 1
                    
                    if game_result["game_id"] in existing_game_ids:
                        # Schedule a new task
                        return
                    
                    # Use the role information from the result
                    tiger_config_str = game_result["tiger_config_str"]
                    goat_config_str = game_result["goat_config_str"]
                    is_config1_tiger = game_result["is_config1_tiger"]
                    
                    # Get configurations to access details
                    tiger_config = json.loads(tiger_config_str)
                    goat_config = json.loads(goat_config_str)
                    
                    # Generate readable summaries
                    tiger_summary = get_config_summary(tiger_config)
                    goat_summary = get_config_summary(goat_config)
                    
                    # Determine the winner and create a concise message
                    winner = game_result["winner"].upper()
                    moves = game_result["moves"]
                    goats_captured = game_result["goats_captured"]
                    game_id = games_played  # Use a simple counter as game ID
                    
                    # Create victory message (one line, concise)
                    if winner == "TIGER":
                        victory_emoji = "🐯"
                        victory_msg = f"#{game_id} COMPLETED: {victory_emoji} {tiger_summary} defeated {goat_summary} ({moves} moves, {goats_captured} captures)"
                    elif winner == "GOAT":
                        victory_emoji = "🐐"
                        victory_msg = f"#{game_id} COMPLETED: {victory_emoji} {goat_summary} defeated {tiger_summary} ({moves} moves)"
                    else:
                        victory_emoji = "🤝"
                        victory_msg = f"#{game_id} COMPLETED: {victory_emoji} DRAW between {tiger_summary} and {goat_summary} ({moves} moves)"
                    
                    print(victory_msg)
                    
                    # Update matchup counts
                    for idx, data in matchup_game_counts.items():
                        if (data['config1_str'] == (tiger_config_str if is_config1_tiger else goat_config_str) and
                            data['config2_str'] == (goat_config_str if is_config1_tiger else tiger_config_str)):
                            # Decrement the in-progress counter first
                            if is_config1_tiger:
                                matchup_game_counts[idx]['tiger1_goat2_in_progress'] -= 1
                                matchup_game_counts[idx]['tiger1_goat2'] += 1
                            else:
                                matchup_game_counts[idx]['tiger2_goat1_in_progress'] -= 1
                                matchup_game_counts[idx]['tiger2_goat1'] += 1
                            matchup_game_counts[idx]['total'] += 1
                            break
                    
                    # Prepare row for CSV
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
                    
                    # Write to CSV
                    writer.writerow(row)
                    csvfile.flush()
                    games_saved += 1
                    existing_game_ids.add(game_result["game_id"])
                    
                    # Sync to Google Sheets
                    self.sheets_sync.add_row(row, self.csv_headers)
                    
                    # Update progress occasionally
                    current_time = time.time()
                    if current_time - last_progress_update > 5:
                        elapsed_minutes = (current_time - simulation_start_time) / 60
                        remaining_minutes = (end_time - current_time) / 60
                        
                        print(f"\n==== TOURNAMENT PROGRESS ====")
                        print(f"🎮 Games completed: {games_played} games played, {games_saved} saved to records")
                        print(f"⏱️ Time: {elapsed_minutes:.1f} minutes elapsed, {remaining_minutes:.1f} minutes remaining")
                        
                        # Calculate game distribution stats
                        counts = [data['total'] for data in matchup_game_counts.values()]
                        min_count = min(counts) if counts else 0
                        max_count = max(counts) if counts else 0
                        avg_count = sum(counts) / len(counts) if counts else 0
                        
                        print(f"📊 Game distribution: min={min_count}, max={max_count}, avg={avg_count:.1f}")
                        print(f"🔄 Active matches: {active_tasks}")
                        print(f"============================\n")
                        last_progress_update = current_time
            
            def on_error(error):
                """Handle errors in game execution."""
                nonlocal active_tasks
                with lock:
                    active_tasks -= 1
                    print(f"Error in game: {error}")
                    
                    # Since we can't easily identify which specific task failed,
                    # we'll make a conservative adjustment to maintain balance:
                    # Find the matchup with the highest in-progress count and decrement it
                    max_in_progress = 0
                    max_idx = None
                    tiger1_higher = False
                    
                    for idx, data in matchup_game_counts.items():
                        tiger1_count = data['tiger1_goat2_in_progress']
                        tiger2_count = data['tiger2_goat1_in_progress']
                        
                        if tiger1_count > max_in_progress:
                            max_in_progress = tiger1_count
                            max_idx = idx
                            tiger1_higher = True
                        
                        if tiger2_count > max_in_progress:
                            max_in_progress = tiger2_count
                            max_idx = idx
                            tiger1_higher = False
                    
                    # Decrement the highest in-progress counter
                    if max_idx is not None and max_in_progress > 0:
                        if tiger1_higher:
                            matchup_game_counts[max_idx]['tiger1_goat2_in_progress'] -= 1
                        else:
                            matchup_game_counts[max_idx]['tiger2_goat1_in_progress'] -= 1
            
            def schedule_task(pool, task):
                """Schedule a single task and track it."""
                nonlocal active_tasks
                with lock:
                    if time.time() < end_time:
                        config1, config2, is_config1_tiger, config1_str, config2_str = task
                        tiger_config = config1 if is_config1_tiger else config2
                        goat_config = config2 if is_config1_tiger else config1
                        
                        # Generate readable summaries for the matchup
                        tiger_summary = get_config_summary(tiger_config)
                        goat_summary = get_config_summary(goat_config)
                        
                        # Display info about the new matchup being scheduled
                        match_id = active_tasks + 1  # Simple ID for the new match
                        match_msg = f"#{match_id} STARTING: 🐯 {tiger_summary} vs 🐐 {goat_summary}"
                        print(match_msg)
                        
                        # Schedule the task
                        pool.apply_async(
                            self._run_game_wrapper,
                            args=(task,),
                            callback=on_game_complete,
                            error_callback=on_error
                        )
                        active_tasks += 1
                        return True
                return False
            
            # Create pool and run games
            try:
                with mp.Pool(processes=parallel_games) as pool:
                    # Get initial tasks to schedule
                    initial_tasks = get_next_tasks(parallel_games)
                    print(f"\n🎬 TOURNAMENT STARTING! Scheduling initial batch of {len(initial_tasks)} games...")
                    print(f"⚔️ Let the games begin! ⚔️\n")
                    
                    # Schedule initial tasks
                    for task in initial_tasks:
                        schedule_task(pool, task)
                    
                    # Flag to track whether we've shown the time limit message
                    time_limit_message_shown = False
                    
                    # Main simulation loop
                    while active_tasks > 0 and time.time() < end_time:
                        # Get tasks to replace completed ones
                        if active_tasks < parallel_games:
                            tasks_needed = parallel_games - active_tasks
                            new_tasks = get_next_tasks(tasks_needed)
                            
                            for task in new_tasks:
                                if not schedule_task(pool, task):
                                    break
                        
                        # Sleep briefly to avoid CPU spinning
                        time.sleep(0.1)
                        
                        # If we're close to time limit, notify once
                        if time.time() + 60 > end_time and active_tasks > 0 and not time_limit_message_shown:
                            print(f"Time limit approaching. Waiting for {active_tasks} active tasks to complete.")
                            time_limit_message_shown = True
                    
                    # Make sure we wait for all tasks to complete
                    pool.close()
                    pool.join()
                    
            except Exception as e:
                print(f"Error during simulation: {e}")
                import traceback
                traceback.print_exc()
            
            # Summary
            elapsed_minutes = (time.time() - simulation_start_time) / 60
            print(f"\n🏆 TOURNAMENT COMPLETE! 🏆")
            print(f"⏱️ Total time: {elapsed_minutes:.1f} minutes")
            print(f"🎮 Total games played: {games_played}")
            print(f"💾 Total games saved: {games_saved}")
            
            # Calculate final game distribution stats
            counts = [data['total'] for data in matchup_game_counts.values()]
            min_count = min(counts) if counts else 0
            max_count = max(counts) if counts else 0
            avg_count = sum(counts) / len(counts) if counts else 0
            
            print(f"📊 Final game distribution: min={min_count}, max={max_count}, avg={avg_count:.1f}")
            print(f"📁 Results saved to: {output_file}")
            
            # Final sync to Google Sheets
            self.sheets_sync.sync(force=True)
        
        return output_file 