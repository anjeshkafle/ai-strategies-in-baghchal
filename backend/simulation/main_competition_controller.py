"""
Controller for running the main competition between MCTS and Minimax agents.
"""
import os
import sys
import time
import json
import csv
import itertools
import multiprocessing as mp
import pandas as pd
import random
import logging
import traceback
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.game_state import GameState
from models.mcts_agent import MCTSAgent
from models.minimax_agent import MinimaxAgent
from simulation.game_runner import GameRunner
from simulation.google_sheets_sync import GoogleSheetsSync

class MainCompetitionController:
    """
    Controller for running the main competition between MCTS and Minimax agents.
    """
    
    def __init__(self, mcts_configs_path: str, minimax_depths: List[int],
                 output_dir: str, sheets_url: str = None, batch_size: int = 50,
                 max_time_per_move: int = 1):
        """
        Initialize the competition controller.
        
        Args:
            mcts_configs_path: Path to CSV containing top MCTS configurations
            minimax_depths: List of depths for Minimax agent
            output_dir: Directory to store results
            sheets_url: Google Sheets Web App URL for syncing results
            batch_size: Batch size for Google Sheets sync
            max_time_per_move: Maximum time in seconds allowed per move
        """
        # Store parameters
        self.mcts_configs_path = mcts_configs_path
        self.minimax_depths = minimax_depths
        self.output_dir = output_dir
        self.max_time_per_move = max_time_per_move
        
        # Set up logger first
        self._setup_logger()
        
        # Load configuration from run_main_competition.py
        self.main_config = {}
        main_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "simulation", "main_competition_config.json")
        if os.path.exists(main_config_path):
            with open(main_config_path, 'r') as f:
                self.main_config = json.load(f)
        
        # Use passed parameter over config file
        if max_time_per_move is not None:
            self.max_time_per_move = max_time_per_move
        else:
            self.max_time_per_move = self.main_config.get('max_time_per_move', 1)
        
        # Set up Google Sheets sync if enabled
        self.sheets_sync = None
        if sheets_url:
            try:
                self.sheets_sync = GoogleSheetsSync(
                    webapp_url=sheets_url,
                    batch_size=batch_size,
                    output_dir=output_dir
                )
                self.logger.info(f"Google Sheets sync enabled (batch size: {batch_size})")
            except Exception as e:
                self.logger.error(f"Failed to initialize Google Sheets sync: {e}")
                print(f"Google Sheets sync failed to initialize: {e}")
                self.sheets_sync = None
                
        # Load configurations
        self.mcts_configs = self._load_mcts_configs()
        self.minimax_configs = self._create_minimax_configs()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("=== Starting Main Competition between MCTS and Minimax ===")
    
    def _setup_logger(self):
        """Set up logging to file and console."""
        logger = logging.getLogger('main_competition')
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if logger.handlers:
            for handler in logger.handlers:
                logger.removeHandler(handler)
        
        # Create log directory if needed
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create log file with consistent name (no timestamp)
        log_file = os.path.join(output_dir, "competition.log")
        
        # File handler for detailed logging - append mode
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler for basic output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Add a separator in the log file to distinguish runs
        with open(log_file, 'a') as f:
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"COMPETITION RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        self.logger = logger
        return logger
    
    def _load_mcts_configs(self) -> List[Dict]:
        """
        Load top MCTS configurations from CSV file.
        
        Returns:
            List of MCTS configurations
        """
        if not os.path.exists(self.mcts_configs_path):
            raise FileNotFoundError(f"MCTS configurations file not found: {self.mcts_configs_path}")
        
        mcts_configs = []
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.mcts_configs_path)
            
            # Ensure required columns exist
            required_columns = ['config_id', 'rollout_policy', 'rollout_depth', 'exploration_weight']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in MCTS configurations file")
            
            # Convert each row to a configuration dict
            for _, row in df.iterrows():
                config = {
                    'algorithm': 'mcts',
                    'config_id': row['config_id'],
                    'rollout_policy': row['rollout_policy'],
                    'rollout_depth': int(row['rollout_depth']),
                    'exploration_weight': float(row['exploration_weight']),
                    # Use max_time_per_move from config rather than hardcoded value
                    'max_time_seconds': self.max_time_per_move
                }
                
                # Add optional parameters if present
                if 'guided_strictness' in row and not pd.isna(row['guided_strictness']):
                    config['guided_strictness'] = float(row['guided_strictness'])
                
                if 'iterations' in row and not pd.isna(row['iterations']):
                    config['iterations'] = int(row['iterations'])
                
                mcts_configs.append(config)
            
            self.logger.info(f"Loaded {len(mcts_configs)} MCTS configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading MCTS configurations: {e}")
            raise
        
        return mcts_configs
    
    def _create_minimax_configs(self) -> List[Dict]:
        """
        Create Minimax configurations with specified depths.
        
        Returns:
            List of Minimax configurations
        """
        configs = []
        self.logger.info(f"Creating Minimax configurations with depths: {self.minimax_depths}")
        
        for depth in self.minimax_depths:
            configs.append({
                'algorithm': 'minimax',
                'config_id': f'minimax-d{depth}-tuned',
                'max_depth': depth,
                'use_tuned_params': True,
                # Use max_time_per_move from config rather than hardcoded value
                'max_time_seconds': self.max_time_per_move
            })
        
        self.logger.info(f"Created {len(configs)} Minimax configurations")
        for i, config in enumerate(configs):
            self.logger.info(f"  Minimax #{i+1}: {config['config_id']}")
            
        return configs
    
    def run_competition(self, games_per_matchup: int = None, 
                     max_simulation_time: int = 60,
                     parallel_games: int = None,
                     output_file: str = None) -> str:
        """
        Run the competition between MCTS and Minimax agents.
        
        Args:
            games_per_matchup: Number of games to play per matchup (if None, will run until time limit)
            max_simulation_time: Maximum simulation time in minutes
            parallel_games: Number of games to run in parallel
            output_file: Optional path to existing output file to resume from
            
        Returns:
            Path to the results CSV file
        """
        # Set default parallel games if not specified
        if parallel_games is None:
            parallel_games = max(1, mp.cpu_count() - 1)
            
        # Ensure we don't try to use more processes than available
        parallel_games = min(parallel_games, mp.cpu_count())
        
        # Calculate end time
        simulation_start_time = time.time()
        end_time = simulation_start_time + (max_simulation_time * 60)
        
        if games_per_matchup is None:
            # If games_per_matchup is not specified, set to a large number
            # as we'll be limited by time instead
            self.logger.info(f"No games_per_matchup specified - will run until time limit of {max_simulation_time} minutes")
            games_per_matchup = float('inf')  # Use infinity to indicate time-limited run
            display_games_per_matchup = "∞"  # Display infinity symbol for time-limited runs
        else:
            display_games_per_matchup = str(games_per_matchup)
        
        self.logger.info(f"Starting competition with max_time={max_simulation_time}min, games_per_matchup={display_games_per_matchup}, parallel_games={parallel_games}")
        
        # Generate all matchups (each MCTS config vs each Minimax config)
        all_matchups = []
        
        # Create matchup pairs - each pair represents the same agents with sides swapped
        matchup_pairs = []
        
        for mcts_config in self.mcts_configs:
            for minimax_config in self.minimax_configs:
                # Create a unique identifier for this matchup pair
                pair_id = len(matchup_pairs)
                
                # MCTS as Tiger, Minimax as Goat
                idx1 = len(all_matchups)
                all_matchups.append((mcts_config, minimax_config, "TIGER", "GOAT"))
                
                # Minimax as Tiger, MCTS as Goat
                idx2 = len(all_matchups)
                all_matchups.append((minimax_config, mcts_config, "TIGER", "GOAT"))
                
                # Add this pair to our pairs list
                matchup_pairs.append((pair_id, idx1, idx2))
                
                # Log with emojis and better formatting
                mcts_summary = self._get_config_summary(mcts_config)
                minimax_summary = self._get_config_summary(minimax_config)
                self.logger.info(f"🎭 Created matchup pair {pair_id}: 🐯🐐 {mcts_summary} vs {minimax_summary}")
        
        total_matchups = len(all_matchups)
        if games_per_matchup == float('inf'):
            # For time-limited runs, set a very large number for display
            total_games = 1000000  # Just a large number for progress percentage
            display_total_games = "∞"  # Display infinity symbol
        else:
            total_games = total_matchups * games_per_matchup
            display_total_games = str(total_games)
        
        self.logger.info(f"Competition setup complete:")
        self.logger.info(f"  MCTS configurations: {len(self.mcts_configs)}")
        self.logger.info(f"  Minimax configurations: {len(self.minimax_configs)}")
        self.logger.info(f"  Total matchups: {total_matchups}")
        self.logger.info(f"  Total matchup pairs: {len(matchup_pairs)}")
        self.logger.info(f"  Games per matchup: {display_games_per_matchup}")
        self.logger.info(f"  Total games target: {display_total_games} (running until time limit: {max_simulation_time} minutes)")
        
        # Print with emojis
        print(f"\n🏆 Main Competition Setup:")
        print(f"  🤖 MCTS configurations: {len(self.mcts_configs)}")
        print(f"  🧠 Minimax configurations: {len(self.minimax_configs)}")
        print(f"  ⚔️ Total matchups: {total_matchups}")
        print(f"  🔄 Total matchup pairs: {len(matchup_pairs)}")
        print(f"  🎮 Games per matchup: {display_games_per_matchup}")
        print(f"  📊 Total games target: {display_total_games}")
        print(f"  🔄 Parallel games: {parallel_games}")
        print(f"  ⏱️ Max simulation time: {max_simulation_time} minutes")
        print(f"  🎯 Target: Full side balance for all matchups")
        
        # Create or use existing output file
        if output_file is None or not os.path.exists(output_file):
            # Create a new output file with a fixed name (no timestamp)
            output_file = os.path.join(self.output_dir, "main_competition.csv")
            self.logger.info(f"Creating new output file: {output_file}")
            
            # CSV header fields
            csv_fields = [
                'game_id', 'winner', 'reason', 'moves', 'game_duration',
                'avg_tiger_move_time', 'avg_goat_move_time',
                'first_capture_move', 'goats_captured', 'phase_transition_move',
                'move_history', 
                'tiger_algorithm', 'tiger_config', 
                'goat_algorithm', 'goat_config'
            ]
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
                writer.writeheader()
        else:
            self.logger.info(f"Resuming from existing file: {output_file}")
        
        # Get existing games from output file
        existing_game_ids = set()
        existing_matchups = {}
        
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_game_ids.add(row['game_id'])
                        
                        # Extract tiger/goat configs for matchup tracking
                        tiger_config = json.loads(row['tiger_config'])
                        goat_config = json.loads(row['goat_config'])
                        tiger_str = json.dumps(tiger_config)
                        goat_str = json.dumps(goat_config)
                        
                        # Count this matchup
                        matchup_key = (tiger_str, goat_str)
                        existing_matchups[matchup_key] = existing_matchups.get(matchup_key, 0) + 1
                        
                self.logger.info(f"Found {len(existing_game_ids)} existing games in output file")
                print(f"Found {len(existing_game_ids)} existing games in output file")
            except Exception as e:
                self.logger.error(f"Error reading existing file: {e}")
                print(f"Error reading existing file: {e}")
        
        # Track matchup game counts - initialize with existing games
        matchup_game_counts = {}
        for i, (config1, config2, _, _) in enumerate(all_matchups):
            config1_str = json.dumps(config1)
            config2_str = json.dumps(config2)
            
            # Find which pair this matchup belongs to
            pair_id = None
            for p_id, idx1, idx2 in matchup_pairs:
                if i == idx1 or i == idx2:
                    pair_id = p_id
                    break
            
            # Count existing games for this matchup
            existing_count = existing_matchups.get((config1_str, config2_str), 0)
            
            matchup_game_counts[i] = {
                'total': existing_count,  # Start with count from existing file
                'config1': config1,
                'config2': config2,
                'config1_str': config1_str,
                'config2_str': config2_str,
                'pair_id': pair_id
            }
        
        # Run the competition using multiprocessing
        games_played = len(existing_game_ids)  # Start from existing count
        games_saved = len(existing_game_ids)   # Start from existing count
        
        self.logger.info("Starting competition with multiprocessing")
        print(f"\n🎬 COMPETITION STARTING! Using {parallel_games} parallel processes")
        print(f"⚔️ Let the games begin! ⚔️\n")
        
        try:
            # Set up multiprocessing with a lock for shared resource access
            manager = mp.Manager()
            lock = manager.Lock()
            
            # Use a synchronized counter for active tasks
            active_tasks = manager.Value('i', 0)
            
            # Use multiprocessing.Pool for parallelism
            with mp.Pool(processes=parallel_games) as pool:
                # Function to get next task to execute
                def get_next_tasks(num_tasks=1):
                    """
                    Get the next highest priority matchups to execute.
                    Priority order:
                    1. Balance sides for matchup pairs that have an imbalance
                    2. Pick pairs with the fewest total games
                    """
                    tasks = []
                    
                    with lock:
                        # First, calculate pair totals and identify imbalances
                        pair_totals = {}  # pair_id -> total games
                        pair_imbalances = {}  # pair_id -> (side1_games, side2_games)
                        
                        for matchup_idx, matchup_data in matchup_game_counts.items():
                            pair_id = matchup_data['pair_id']
                            
                            # Initialize if first time seeing this pair
                            if pair_id not in pair_totals:
                                pair_totals[pair_id] = 0
                                pair_imbalances[pair_id] = {}
                            
                            # Add to pair total
                            pair_totals[pair_id] += matchup_data['total']
                            
                            # Track games for this specific matchup in the pair
                            pair_imbalances[pair_id][matchup_idx] = matchup_data['total']
                        
                        # Find pairs with imbalances (one side played more than the other)
                        imbalanced_pairs = []
                        for pair_id, matchups in pair_imbalances.items():
                            if len(matchups) == 2:  # Should always be true
                                idx1, idx2 = matchups.keys()
                                side1_games = matchups[idx1]
                                side2_games = matchups[idx2]
                                
                                # If sides are imbalanced and not complete
                                if side1_games != side2_games and (side1_games < games_per_matchup or side2_games < games_per_matchup):
                                    imbalanced_pairs.append((pair_id, idx1, idx2, side1_games, side2_games))
                        
                        # Sort pairs by largest imbalance
                        imbalanced_pairs.sort(key=lambda x: abs(x[3] - x[4]), reverse=True)
                        
                        # First priority: balance sides
                        for pair_id, idx1, idx2, side1_games, side2_games in imbalanced_pairs:
                            # If we've collected enough tasks, break
                            if len(tasks) >= num_tasks:
                                break
                                
                            # Determine which side needs more games
                            if side1_games < side2_games and (games_per_matchup == float('inf') or side1_games < games_per_matchup):
                                matchup_idx = idx1
                            elif side2_games < side1_games and (games_per_matchup == float('inf') or side2_games < games_per_matchup):
                                matchup_idx = idx2
                            else:
                                continue  # Skip if both sides at max games or balanced
                            
                            # Get matchup details
                            matchup = all_matchups[matchup_idx]
                            tiger_config, goat_config, tiger_role, goat_role = matchup
                            
                            # Create a unique seed for reproducibility
                            seed = hash(f"{matchup_idx}_{matchup_game_counts[matchup_idx]['total']}") % (2**32)
                            
                            # Create the task with UUID as game_id
                            task = {
                                'matchup_idx': matchup_idx,
                                'tiger_config': tiger_config,
                                'goat_config': goat_config,
                                'tiger_role': tiger_role,
                                'goat_role': goat_role,
                                'seed': seed,
                                'game_id': str(uuid.uuid4())
                            }
                            
                            # Update the count for this matchup
                            matchup_game_counts[matchup_idx]['total'] += 1
                            
                            # Add task to the list
                            tasks.append(task)
                        
                        # If we still need more tasks, pick pairs with fewest total games
                        if len(tasks) < num_tasks:
                            # Sort pairs by total games played
                            sorted_pairs = sorted(pair_totals.items(), key=lambda x: x[1])
                            
                            for pair_id, total in sorted_pairs:
                                # Skip completed pairs - but don't skip if games_per_matchup is infinity
                                if games_per_matchup != float('inf') and total >= games_per_matchup * 2:
                                    continue
                                    
                                # Find the matchups for this pair
                                idx1, idx2 = None, None
                                for matchup_idx, matchup_data in matchup_game_counts.items():
                                    if matchup_data['pair_id'] == pair_id:
                                        if idx1 is None:
                                            idx1 = matchup_idx
                                        else:
                                            idx2 = matchup_idx
                                            break
                                
                                # Make sure both sides have equal games when adding new tasks
                                side1_games = matchup_game_counts[idx1]['total']
                                side2_games = matchup_game_counts[idx2]['total']
                                
                                # Only add if both sides < games_per_matchup (or games_per_matchup is infinity)
                                if games_per_matchup == float('inf') or (side1_games < games_per_matchup and side2_games < games_per_matchup):
                                    # Add tasks for both sides if possible
                                    for idx in [idx1, idx2]:
                                        if games_per_matchup == float('inf') or matchup_game_counts[idx]['total'] < games_per_matchup:
                                            # Get matchup details
                                            matchup = all_matchups[idx]
                                            tiger_config, goat_config, tiger_role, goat_role = matchup
                                            
                                            # Create a unique seed for reproducibility
                                            seed = hash(f"{idx}_{matchup_game_counts[idx]['total']}") % (2**32)
                                            
                                            # Create the task with UUID as game_id
                                            task = {
                                                'matchup_idx': idx,
                                                'tiger_config': tiger_config,
                                                'goat_config': goat_config,
                                                'tiger_role': tiger_role,
                                                'goat_role': goat_role,
                                                'seed': seed,
                                                'game_id': str(uuid.uuid4())
                                            }
                                            
                                            # Update the count for this matchup
                                            matchup_game_counts[idx]['total'] += 1
                                            
                                            # Add task to the list
                                            tasks.append(task)
                                            
                                            # If we've collected enough tasks, break
                                            if len(tasks) >= num_tasks:
                                                break
                                
                                # If we've collected enough tasks, break
                                if len(tasks) >= num_tasks:
                                    break
                    
                    return tasks
                
                # Function to handle completed game result
                def on_game_complete(game_result):
                    """Callback when a game completes - process results and schedule next task."""
                    nonlocal games_played, games_saved, active_tasks
                    
                    with lock:
                        active_tasks.value -= 1
                        games_played += 1
                        
                        # Check if this game ID is already in our list of existing games
                        # This shouldn't happen with UUIDs but check just to be safe
                        if game_result['game_id'] in existing_game_ids:
                            self.logger.warning(f"Duplicate game ID encountered: {game_result['game_id']}")
                        else:
                            existing_game_ids.add(game_result['game_id'])
                        
                        # Save the game result
                        self._save_game_result(game_result, output_file)
                        games_saved += 1
                        
                        # Log completion
                        self.logger.debug(f"Game {game_result['game_id']} completed. Active tasks: {active_tasks.value}")
                
                def on_error(e):
                    """Handle errors in the game execution."""
                    nonlocal active_tasks
                    
                    with lock:
                        active_tasks.value -= 1
                        self.logger.error(f"Error in game: {e}")
                        print(f"❌ Error in game: {e}")
                        
                        # Log error with active task count
                        self.logger.debug(f"Game error. Active tasks: {active_tasks.value}")
                
                def schedule_task(task):
                    """Schedule a single task and keep track of it."""
                    nonlocal active_tasks
                    
                    with lock:
                        if time.time() < end_time:
                            tiger_config = task['tiger_config']
                            goat_config = task['goat_config']
                            
                            # Generate readable summaries for the matchup
                            tiger_summary = self._get_config_summary(tiger_config)
                            goat_summary = self._get_config_summary(goat_config)
                            
                            # Display info about the new matchup being scheduled
                            match_id = task['game_id']  # Use the game ID for the match
                            # Shorten the UUID for display purposes
                            short_id = match_id[:8] if len(match_id) > 8 else match_id
                            match_msg = f"Game {short_id} STARTING: 🐯 {tiger_summary} vs 🐐 {goat_summary}"
                            print(match_msg)
                            
                            # Increment active tasks before scheduling to prevent race condition
                            active_tasks.value += 1
                            
                            # Schedule the task
                            pool.apply_async(
                                self._play_game, 
                                (task['tiger_config'], task['goat_config'], 
                                 task['tiger_role'], task['goat_role'],
                                 task['seed'], task['game_id']),
                                callback=on_game_complete,
                                error_callback=on_error
                            )
                            
                            # Log scheduling
                            self.logger.debug(f"Game {short_id} scheduled. Active tasks: {active_tasks.value}")
                            return True
                    return False
                
                # Start initial batch of tasks
                initial_tasks = get_next_tasks(parallel_games)
                self.logger.info(f"Starting initial batch of {len(initial_tasks)} games")
                
                # Schedule the initial tasks - don't set active_tasks.value directly
                active_tasks.value = 0  # Reset to ensure accurate count
                for task in initial_tasks:
                    schedule_task(task)
                
                # Flag to track whether we've shown the time limit message
                time_limit_message_shown = False
                last_progress_update = time.time()
                last_games_played = games_played  # Track last number of completed games
                
                # Main simulation loop
                while active_tasks.value > 0 and time.time() < end_time:
                    # Get tasks to replace completed ones
                    if active_tasks.value < parallel_games:
                        tasks_needed = parallel_games - active_tasks.value
                        new_tasks = get_next_tasks(tasks_needed)
                        
                        for task in new_tasks:
                            if not schedule_task(task):
                                break
                    
                    # Sleep briefly to avoid CPU spinning
                    time.sleep(0.1)
                    
                    # If we're close to time limit, notify once
                    if time.time() + 60 > end_time and active_tasks.value > 0 and not time_limit_message_shown:
                        print(f"\n⏰ Time limit approaching. Waiting for {active_tasks.value} active tasks to complete.")
                        time_limit_message_shown = True
                        
                    # Update progress periodically (every 10 seconds) OR when a game completes
                    progress_time_interval = time.time() - last_progress_update >= 10
                    games_completed = games_played > last_games_played
                    
                    if progress_time_interval or games_completed:
                        # Log progress update
                        elapsed_min = (time.time() - simulation_start_time) / 60
                        
                        # Handle progress display differently for time-limited runs
                        if games_per_matchup == float('inf'):
                            # Show completed games without percentage for infinity case
                            progress_str = f"{games_played} games completed - {elapsed_min:.1f}min elapsed"
                            if games_played > 0:
                                # Estimate games per minute
                                games_per_min = games_played / elapsed_min
                                est_games_remaining = games_per_min * (max_simulation_time - elapsed_min)
                                progress_str += f", ~{est_games_remaining:.0f} more games possible in remaining time"
                            self.logger.info(progress_str)
                        else:
                            # Normal percentage-based progress for fixed games_per_matchup
                            progress_pct = games_played/total_games*100
                            if games_played > 0:
                                est_total_min = (elapsed_min / games_played) * total_games
                                est_remaining_min = est_total_min - elapsed_min
                                self.logger.info(f"Progress: {games_played}/{total_games} games ({progress_pct:.1f}%) - {elapsed_min:.1f}min elapsed, ~{est_remaining_min:.1f}min remaining")
                            else:
                                self.logger.info(f"Progress: {games_played}/{total_games} games ({progress_pct:.1f}%) - {elapsed_min:.1f}min elapsed")
                        
                        # Only print detailed progress when games are completed or at first update
                        if games_completed or (progress_time_interval and last_games_played == 0):
                            # Print progress to console in a more attractive format with emojis
                            print(f"\n==== COMPETITION PROGRESS ====")
                            if games_per_matchup == float('inf'):
                                # Show completed games without percentage for infinity case
                                print(f"🎮 Games completed: {games_played}")
                                print(f"⏱️ Time: {elapsed_min:.1f} minutes elapsed, {max_simulation_time - elapsed_min:.1f} minutes remaining")
                                if games_played > 0:
                                    games_per_min = games_played / elapsed_min
                                    print(f"📊 Rate: {games_per_min:.1f} games/minute")
                            else:
                                # Normal percentage-based progress for fixed games_per_matchup
                                print(f"🎮 Games completed: {games_played}/{total_games} ({progress_pct:.1f}%)")
                                print(f"⏱️ Time: {elapsed_min:.1f} minutes elapsed", end="")
                                if games_played > 0:
                                    print(f", ~{est_remaining_min:.1f} minutes remaining")
                                else:
                                    print()
                            
                            # Calculate game distribution stats
                            # Only count completed games, not those just scheduled
                            # Create a count of completed games per matchup
                            completed_counts = {}
                            for idx in matchup_game_counts:
                                # Initialize with 0 completed games (we'll count them below)
                                completed_counts[idx] = 0 
                            
                            # Count completed games from the CSV file
                            if os.path.exists(output_file):
                                try:
                                    with open(output_file, 'r') as f:
                                        reader = csv.DictReader(f)
                                        for row in reader:
                                            # Find the matchup this game belongs to
                                            tiger_config = json.loads(row['tiger_config'])
                                            goat_config = json.loads(row['goat_config'])
                                            
                                            # Find which matchup this corresponds to
                                            for idx, data in matchup_game_counts.items():
                                                match_tiger = (data['config1_str'] == json.dumps(tiger_config))
                                                match_goat = (data['config2_str'] == json.dumps(goat_config))
                                                if match_tiger and match_goat:
                                                    completed_counts[idx] += 1
                                                    break
                                except Exception as e:
                                    self.logger.error(f"Error reading output file for stats: {e}")
                            
                            # Use completed counts for statistics
                            counts = list(completed_counts.values())
                            min_count = min(counts) if counts else 0
                            max_count = max(counts) if counts else 0
                            avg_count = sum(counts) / len(counts) if counts else 0
                            
                            print(f"📊 Game distribution: min={min_count}, max={max_count}, avg={avg_count:.1f}")
                            print(f"🔄 Active games: {active_tasks.value}")  # Changed from "Active matches" to "Active games"
                            print(f"============================")
                        
                        last_progress_update = time.time()
                        last_games_played = games_played  # Update last games played
                
                # Make sure we wait for all tasks to complete
                pool.close()
                pool.join()
                
            # Print summary
            elapsed_minutes = (time.time() - simulation_start_time) / 60
            self.logger.info(f"Competition complete!")
            self.logger.info(f"Total time: {elapsed_minutes:.1f} minutes")
            self.logger.info(f"Total games played: {games_played}/{total_games} ({games_played/total_games*100:.1f}%)")
            self.logger.info(f"Results saved to: {output_file}")
            
            print(f"\n\n🏆 COMPETITION COMPLETE! 🏆")
            print(f"⏱️ Total time: {elapsed_minutes:.1f} minutes")
            print(f"🎮 Total games played: {games_played}/{total_games} ({games_played/total_games*100:.1f}%)")
            print(f"📁 Results saved to: {output_file}")
            
            # Verify side balance
            pair_stats = {}
            for matchup_idx, matchup_data in matchup_game_counts.items():
                pair_id = matchup_data['pair_id']
                if pair_id not in pair_stats:
                    pair_stats[pair_id] = {}
                
                pair_stats[pair_id][matchup_idx] = matchup_data['total']
            
            imbalanced_pairs = []
            for pair_id, stats in pair_stats.items():
                if len(stats) == 2:
                    idx1, idx2 = stats.keys()
                    if stats[idx1] != stats[idx2]:
                        imbalanced_pairs.append(f"Pair {pair_id}: {stats[idx1]} vs {stats[idx2]}")
            
            if imbalanced_pairs:
                self.logger.warning("Side imbalances detected:")
                for imbalance in imbalanced_pairs:
                    self.logger.warning(f"  {imbalance}")
                
                print("\n⚠️ WARNING: Side imbalances detected:")
                for imbalance in imbalanced_pairs:
                    print(f"  {imbalance}")
            else:
                self.logger.info("All matchup pairs have balanced sides")
                print("\n✅ All matchup pairs have balanced sides")
            
            # Final sync with Google Sheets
            if self.sheets_sync:
                self.logger.info("Final sync to Google Sheets")
                self.sheets_sync.sync(force=True)
                
        except KeyboardInterrupt:
            self.logger.warning("Competition interrupted by user")
            print("\n\nInterrupted by user. Saving progress...")
            
            # Final sync with Google Sheets
            if self.sheets_sync:
                self.logger.info("Final sync to Google Sheets after interruption")
                self.sheets_sync.sync(force=True)
            
            # Print summary
            elapsed_minutes = (time.time() - simulation_start_time) / 60
            self.logger.info(f"Interrupted after {elapsed_minutes:.1f} minutes")
            self.logger.info(f"Games played: {games_played}/{total_games} ({games_played/total_games*100:.1f}%)")
            self.logger.info(f"Results saved to: {output_file}")
            
            print(f"\n🛑 COMPETITION INTERRUPTED! 🛑")
            print(f"⏱️ Total time: {elapsed_minutes:.1f} minutes")
            print(f"🎮 Total games played: {games_played}/{total_games} ({games_played/total_games*100:.1f}%)")
            print(f"📁 Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error during competition: {e}")
            self.logger.error(traceback.format_exc())
            
            print(f"\n\n❌ Error during competition: {e}")
            import traceback
            traceback.print_exc()
        
        return output_file
    
    def _play_game(self, tiger_config: Dict, goat_config: Dict, 
                  tiger_role: str, goat_role: str,
                  seed: int, game_id: str) -> Dict:
        """
        Play a single game between two agents.
        
        Args:
            tiger_config: Configuration for the Tiger agent
            goat_config: Configuration for the Goat agent
            tiger_role: Role of the Tiger agent (TIGER/GOAT)
            goat_role: Role of the Goat agent (TIGER/GOAT)
            seed: Random seed for the game
            game_id: Unique ID for the game
            
        Returns:
            Dictionary with game results
        """
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Generate readable summaries for both agents
        tiger_summary = self._get_config_summary(tiger_config)
        goat_summary = self._get_config_summary(goat_config)
        
        # Create short ID for logging
        short_id = game_id[:8] if len(game_id) > 8 else game_id
        
        # Log the game start with emojis and readable agent descriptions
        self.logger.info(f"🎮 Game {short_id} STARTING: 🐯 {tiger_summary} vs 🐐 {goat_summary} (seed: {seed})")
        
        # Create game runner with agent configurations
        # This approach uses the GameRunner class which handles threefold repetition detection
        runner = GameRunner(tiger_config, goat_config)
        
        # Run the game
        result = runner.run_game()
        
        # Update the result with our specific game ID
        result['game_id'] = game_id
        
        # Create more descriptive result message with emojis based on outcome
        winner = result['winner']
        moves = result['moves']
        goats_captured = result['goats_captured']
        
        if winner == "TIGER":
            victory_emoji = "🐯"
            victory_msg = f"Game {short_id} COMPLETED: {victory_emoji} {tiger_summary} defeated {goat_summary} ({moves} moves, {goats_captured} captures)"
        elif winner == "GOAT":
            victory_emoji = "🐐"
            victory_msg = f"Game {short_id} COMPLETED: {victory_emoji} {goat_summary} defeated {tiger_summary} ({moves} moves)"
        else:
            victory_emoji = "🤝"
            reason = result['reason']
            if reason == "THREEFOLD_REPETITION":
                reason_msg = "threefold repetition"
            else:
                reason_msg = reason.lower().replace('_', ' ')
            victory_msg = f"Game {short_id} COMPLETED: {victory_emoji} DRAW between {tiger_summary} and {goat_summary} ({moves} moves, {reason_msg})"
        
        # Log and print the result
        self.logger.info(victory_msg)
        print(victory_msg)
        
        # Create result dictionary matching our expected format
        game_result = {
            'game_id': game_id,
            'winner': result['winner'],
            'reason': result['reason'],
            'moves': result['moves'],
            'game_duration': result['game_duration'],
            'avg_tiger_move_time': result['avg_tiger_move_time'],
            'avg_goat_move_time': result['avg_goat_move_time'],
            'first_capture_move': result['first_capture_move'] if result['first_capture_move'] is not None else -1,
            'goats_captured': result['goats_captured'],
            'phase_transition_move': result['phase_transition_move'] if result['phase_transition_move'] is not None else -1,
            'move_history': result['move_history'],
            'tiger_algorithm': tiger_config['algorithm'],
            'tiger_config': json.dumps(tiger_config),
            'goat_algorithm': goat_config['algorithm'],
            'goat_config': json.dumps(goat_config)
        }
        
        return game_result
    
    def _save_game_result(self, game_result: Dict, output_file: str) -> None:
        """
        Save game result to CSV file.
        
        Args:
            game_result: Dictionary with game results
            output_file: Path to output CSV file
        """
        # Append to CSV file
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=game_result.keys())
            writer.writerow(game_result)
            
        # If Google Sheets sync is enabled, add to queue
        if self.sheets_sync:
            # Pass the headers to the add_row method
            result = self.sheets_sync.add_row(game_result, list(game_result.keys()))
            if result:
                self.logger.info("Triggered sync to Google Sheets (buffer reached batch size)")
            
        # Log the saved game (basic info only)
        game_id = game_result['game_id']
        self.logger.debug(f"Saved game {game_id} to CSV file")
    
    def _create_agent(self, config: Dict) -> Any:
        """
        Create an agent from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Agent instance
        """
        algorithm = config['algorithm'].lower()
        
        if algorithm == 'mcts':
            # Create MCTS agent - only constrained by time, not iterations
            return MCTSAgent(
                rollout_policy=config.get('rollout_policy', 'lightweight'),
                max_rollout_depth=config.get('rollout_depth', 6),
                exploration_weight=config.get('exploration_weight', 1.414),
                guided_strictness=config.get('guided_strictness', 0.8),
                max_time_seconds=config.get('max_time_seconds', 10),
                iterations=None  # Don't constrain by iterations
            )
        elif algorithm == 'minimax':
            # Create Minimax agent - only constrained by depth, not time
            return MinimaxAgent(
                max_depth=config.get('max_depth', 5),
                max_time_seconds=None,  # Don't constrain by time
                randomize_equal_moves=True,
                useTunedParams=config.get('use_tuned_params', True)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
    def _get_config_summary(self, config: Dict) -> str:
        """
        Generate a short readable summary of an agent configuration.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            Human-readable summary string
        """
        algorithm = config.get('algorithm', '').lower()
        
        if algorithm == 'mcts':
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
            
        elif algorithm == 'minimax':
            # For Minimax, highlight the depth and whether tuned params are used
            depth = config.get('max_depth', '?')
            tuned = config.get('use_tuned_params', True)
            
            return f"Minimax-d{depth}{'-tuned' if tuned else ''}"
            
        else:
            return config.get('config_id', 'Unknown') 