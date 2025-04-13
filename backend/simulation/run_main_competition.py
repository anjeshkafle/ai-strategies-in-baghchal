#!/usr/bin/env python3
"""
Run the main competition between the best MCTS configuration and Minimax.

This script runs a large number of games between the best MCTS configuration
(which you must provide) and Minimax agents at different depths.

It also handles interruptions automatically by resuming from the most recent
file, ensuring that all games will be completed even if execution is terminated
and restarted multiple times.

Example usage:
    # Run with default settings
    python run_main_competition.py --policy lightweight --iterations 15000 --depth 6

    # Run with custom settings and number of games
    python run_main_competition.py --policy lightweight --iterations 20000 --depth 4 --games 500
    
    # Force creation of a new file instead of resuming
    python run_main_competition.py --policy lightweight --iterations 15000 --depth 6 --new
"""

import os
import sys
import argparse
import json
import glob
from datetime import datetime

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.simulation.simulation_controller import SimulationController

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run main competition between best MCTS and Minimax")
    
    parser.add_argument("--policy", type=str, required=True, choices=["random", "lightweight", "guided"],
                        help="Best MCTS rollout policy")
    
    parser.add_argument("--iterations", type=int, required=True,
                        help="Best MCTS iteration count")
    
    parser.add_argument("--depth", type=int, required=True,
                        help="Best MCTS rollout depth")
    
    parser.add_argument("--exploration_weight", type=float, default=1.0,
                        help="Exploration weight for UCB formula (default: 1.0)")
    
    parser.add_argument("--guided_strictness", type=float, default=0.8,
                        help="Guided strictness for rollouts (default: 0.8)")
    
    parser.add_argument("--minimax_depths", nargs="+", type=int, default=None,
                        help="Minimax depths to test (space-separated list, default: from config)")
    
    parser.add_argument("--games", type=int, default=None,
                        help="Number of games per matchup (default: from config)")
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: from config)")
    
    parser.add_argument("--new", action="store_true",
                        help="Force creation of a new file instead of resuming")
    
    parser.add_argument("--sheets_url", type=str, default=None,
                       help="Google Sheets web app URL for syncing results")
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for Google Sheets sync (default: from config)")
    
    return parser.parse_args()

def find_most_recent_competition_file(output_dir):
    """
    Find the most recent main competition file.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        The filename of the most recent competition file or None if not found
    """
    pattern = os.path.join(output_dir, "main_competition", "main_competition_*.csv")
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        return None
        
    # Get the most recent file by modification time
    return os.path.basename(max(matching_files, key=os.path.getmtime))

def main():
    """Run the main competition."""
    args = parse_args()
    
    # Load config
    from backend.simulation.config import load_config
    config = load_config()
    
    # Create the best MCTS configuration
    best_mcts_config = {
        'algorithm': 'mcts',
        'rollout_policy': args.policy,
        'iterations': args.iterations,
        'rollout_depth': args.depth,
        'exploration_weight': args.exploration_weight,
        'guided_strictness': args.guided_strictness
    }
    
    # Use command line arguments if provided, otherwise use values from config
    output_dir = args.output_dir or config.main_competition.output_dir
    minimax_depths = args.minimax_depths or config.main_competition.minimax_depths
    games_per_matchup = args.games or config.main_competition.games_per_matchup
    sheets_url = args.sheets_url or config.sheets_webapp_url
    batch_size = args.batch_size or config.sheets_batch_size
    
    print(f"Starting main competition at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Best MCTS: {args.policy} policy, {args.iterations} iterations, depth {args.depth}")
    print(f"  Exploration weight: {args.exploration_weight}")
    print(f"  Guided strictness: {args.guided_strictness}")
    print(f"  Minimax depths: {minimax_depths}")
    print(f"  Games per matchup: {games_per_matchup}")
    print(f"  Output directory: {output_dir}")
    if sheets_url:
        print(f"  Google Sheets sync: Enabled (URL: {sheets_url}, batch size: {batch_size})")
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
        existing_file = find_most_recent_competition_file(output_dir)
        if existing_file:
            print(f"Found existing competition file: {existing_file}")
            print(f"Resuming from this file...")
        else:
            print("No existing competition file found. Starting fresh.")
    
    output_file = controller.run_main_competition(
        best_mcts_config=best_mcts_config,
        minimax_depths=minimax_depths,
        games_per_matchup=games_per_matchup,
        output_file=existing_file
    )
    
    print(f"Main competition progress saved to: {output_file}")
    print(f"You can resume this competition at any time by running the same command.")

if __name__ == "__main__":
    main() 