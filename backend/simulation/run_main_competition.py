#!/usr/bin/env python3
"""
Run the main competition between the best MCTS configuration and Minimax.

This script runs a large number of games between the best MCTS configuration
(which you must provide) and Minimax agents at different depths.

Example usage:
    # Run with default settings
    python run_main_competition.py --policy lightweight --iterations 15000 --depth 6

    # Run with custom settings and number of games
    python run_main_competition.py --policy lightweight --iterations 20000 --depth 4 --games 500
"""

import os
import sys
import argparse
import json
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
    
    parser.add_argument("--minimax_depths", nargs="+", type=int, default=[4, 5, 6],
                        help="Minimax depths to test (space-separated list, default: 4 5 6)")
    
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of games per matchup (default: 1000)")
    
    parser.add_argument("--output_dir", type=str, default="simulation_results",
                        help="Directory to save results (default: simulation_results)")
    
    return parser.parse_args()

def main():
    """Run the main competition."""
    args = parse_args()
    
    # Create the best MCTS configuration
    best_mcts_config = {
        'algorithm': 'mcts',
        'rollout_policy': args.policy,
        'iterations': args.iterations,
        'rollout_depth': args.depth,
        'exploration_weight': 1.0,
        'guided_strictness': 0.8
    }
    
    print(f"Starting main competition at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Best MCTS: {args.policy} policy, {args.iterations} iterations, depth {args.depth}")
    print(f"  Minimax depths: {args.minimax_depths}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    controller = SimulationController(output_dir=args.output_dir)
    
    output_file = controller.run_main_competition(
        best_mcts_config=best_mcts_config,
        minimax_depths=args.minimax_depths,
        games_per_matchup=args.games
    )
    
    print(f"Main competition complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main() 