#!/usr/bin/env python3
"""
Run the MCTS configuration tournament.

This script runs tournament games between different MCTS configurations
to determine the best settings. It supports parallelization by specifying
start and end indices for the matchups to process.

It also handles interruptions automatically by resuming from the most recent
file for the given index range, ensuring that all games will be completed
even if execution is terminated and restarted multiple times.

Example usage:
    # Run all matchups using config file settings
    python run_mcts_tournament.py

    # Run a subset of matchups (for parallel execution)
    python run_mcts_tournament.py --start 0 --end 50
    python run_mcts_tournament.py --start 50 --end 100
"""

import os
import sys
import argparse
from datetime import datetime

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.simulation.simulation_controller import SimulationController
from backend.simulation.config import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MCTS configuration tournament")
    
    parser.add_argument("--games", type=int, default=None,
                        help="Number of games per matchup (default: from config)")
    
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index of matchups to process (default: 0)")
    
    parser.add_argument("--end", type=int, default=None,
                        help="Ending index of matchups to process (default: all)")
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: from config)")

    parser.add_argument("--new", action="store_true",
                        help="Force creation of a new file instead of resuming")
    
    parser.add_argument("--sheets_url", type=str, default=None,
                       help="Google Sheets web app URL for syncing results (default: from config)")
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for Google Sheets sync (default: from config)")
    
    parser.add_argument("--config", type=str, default="simulation_config.json",
                       help="Path to configuration file (default: simulation_config.json)")

    parser.add_argument("--max_time", type=int, default=None,
                       help="Maximum simulation time in minutes (default: from config)")
    
    parser.add_argument("--parallel", type=int, default=None,
                       help="Number of parallel games to run (default: from config)")
    
    return parser.parse_args()

def main():
    """Run the MCTS tournament."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Use command line arguments if provided, otherwise use values from config
    output_dir = args.output_dir or config.mcts_tournament.output_dir
    sheets_url = args.sheets_url or config.sheets_webapp_url
    batch_size = args.batch_size or config.sheets_batch_size
    max_simulation_time = args.max_time or config.mcts_tournament.max_simulation_time
    parallel_games = args.parallel or config.mcts_tournament.parallel_games
    
    # Get all configurations from config
    mcts_configs = config.mcts_tournament.get_all_configs()
    
    # Get unique values for reporting
    policies = set()
    iterations = set()
    depths = set()
    exploration_weights = set()
    guided_strictness_values = set()
    
    for config_item in mcts_configs:
        policies.add(config_item['rollout_policy'])
        if 'iterations' in config_item:
            iterations.add(config_item['iterations'])
        depths.add(config_item['rollout_depth'])
        exploration_weights.add(config_item['exploration_weight'])
        guided_strictness_values.add(config_item['guided_strictness'])
    
    print(f"Starting MCTS tournament at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Configuration groups: {len(config.mcts_tournament.configurations)}")
    print(f"  Unique policies: {sorted(policies)}")
    if iterations:
        print(f"  Unique iterations: {sorted(iterations)}")
    print(f"  Unique depths: {sorted(depths)}")
    print(f"  Unique exploration weights: {sorted(exploration_weights)}")
    print(f"  Unique guided strictness values: {sorted(guided_strictness_values)}")
    print(f"  Total unique configurations: {len(mcts_configs)}")
    print(f"  Max time per move: {config.mcts_tournament.max_time_per_move} seconds")
    print(f"  Max simulation time: {max_simulation_time} minutes")
    print(f"  Processing matchups from index {args.start} to {args.end if args.end is not None else 'end'}")
    print(f"  Output directory: {output_dir}")
    print(f"  Parallel games: {parallel_games if parallel_games else 'auto'}")
    if sheets_url:
        print(f"  Google Sheets sync: Enabled (URL configured, batch size: {batch_size})")
    else:
        print(f"  Google Sheets sync: Disabled")
    print()
    
    controller = SimulationController(
        output_dir=output_dir,
        google_sheets_url=sheets_url,
        batch_size=batch_size
    )
    
    # Look for existing file to resume from, unless --new is specified
    existing_file = None
    if not args.new:
        existing_file = controller.find_most_recent_tournament_file(args.start, args.end)
        if existing_file:
            print(f"Found existing tournament file: {existing_file}")
            print(f"Resuming from this file...")
        else:
            print("No existing tournament file found. Starting fresh.")
    
    output_file = controller.run_mcts_tournament(
        mcts_configs=mcts_configs,
        max_simulation_time=max_simulation_time,
        start_idx=args.start,
        end_idx=args.end,
        output_file=existing_file,
        parallel_games=parallel_games
    )
    
    print(f"Tournament progress saved to: {output_file}")
    print(f"You can resume this tournament at any time by running the same command.")

if __name__ == "__main__":
    main() 