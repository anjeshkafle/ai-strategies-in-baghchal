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
    # Run all matchups
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MCTS configuration tournament")
    
    parser.add_argument("--games", type=int, default=40,
                        help="Number of games per matchup (default: 40)")
    
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index of matchups to process (default: 0)")
    
    parser.add_argument("--end", type=int, default=None,
                        help="Ending index of matchups to process (default: all)")
    
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: from config)")

    parser.add_argument("--new", action="store_true",
                        help="Force creation of a new file instead of resuming")
    
    parser.add_argument("--sheets_url", type=str, default=None,
                       help="Google Sheets web app URL for syncing results")
    
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for Google Sheets sync (default: 100)")
    
    return parser.parse_args()

def main():
    """Run the MCTS tournament."""
    args = parse_args()
    
    # Load config
    from backend.simulation.config import load_config
    config = load_config()
    
    # Use command line arguments if provided, otherwise use values from config
    output_dir = args.output_dir or config.mcts_tournament.output_dir
    sheets_url = args.sheets_url or config.sheets_webapp_url
    batch_size = args.batch_size or config.sheets_batch_size
    max_simulation_time = config.mcts_tournament.max_simulation_time
    
    # Get all configurations from config
    mcts_configs = config.mcts_tournament.get_all_configs()
    
    print(f"Starting MCTS tournament at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Total configurations: {len(mcts_configs)}")
    print(f"  Max simulation time: {max_simulation_time} minutes")
    print(f"  Games per matchup: {args.games}")
    print(f"  Processing matchups from index {args.start} to {args.end if args.end is not None else 'end'}")
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
        output_file=existing_file
    )
    
    print(f"Tournament progress saved to: {output_file}")
    print(f"You can resume this tournament at any time by running the same command.")

if __name__ == "__main__":
    main() 