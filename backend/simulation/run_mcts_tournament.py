#!/usr/bin/env python3
"""
Run the MCTS configuration tournament.

This script runs tournament games between different MCTS configurations
to determine the best settings. It supports parallelization by specifying
start and end indices for the matchups to process.

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
    
    parser.add_argument("--rollout_policies", nargs="+", default=["random", "lightweight", "guided"],
                        help="Rollout policies to test (space-separated list)")
    
    parser.add_argument("--iterations", nargs="+", type=int, default=[10000, 15000, 20000],
                        help="Iteration counts to test (space-separated list)")
    
    parser.add_argument("--rollout_depths", nargs="+", type=int, default=[4, 6],
                        help="Rollout depths to test (space-separated list)")
    
    parser.add_argument("--games", type=int, default=40,
                        help="Number of games per matchup (default: 40)")
    
    parser.add_argument("--start", type=int, default=0,
                        help="Starting index of matchups to process (default: 0)")
    
    parser.add_argument("--end", type=int, default=None,
                        help="Ending index of matchups to process (default: all)")
    
    parser.add_argument("--output_dir", type=str, default="simulation_results",
                        help="Directory to save results (default: simulation_results)")
    
    return parser.parse_args()

def main():
    """Run the MCTS tournament."""
    args = parse_args()
    
    print(f"Starting MCTS tournament at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Rollout policies: {args.rollout_policies}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Rollout depths: {args.rollout_depths}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Processing matchups from index {args.start} to {args.end if args.end is not None else 'end'}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    controller = SimulationController(output_dir=args.output_dir)
    
    output_file = controller.run_mcts_tournament(
        rollout_policies=args.rollout_policies,
        iterations=args.iterations,
        rollout_depths=args.rollout_depths,
        games_per_matchup=args.games,
        start_idx=args.start,
        end_idx=args.end
    )
    
    print(f"Tournament complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main() 