#!/usr/bin/env python3
"""
Run analysis on MCTS tournament results.

This script analyzes tournament results to determine the best MCTS configurations.

Example usage:
    # Analyze results from latest tournament
    python run_mcts_analysis.py
    
    # Specify input file and output directory
    python run_mcts_analysis.py --input path/to/results.csv --output path/to/output
    
    # Use custom configuration file
    python run_mcts_analysis.py --config path/to/config.json
"""

import os
import sys
import argparse
import glob
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from mcts_analysis import main_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MCTS tournament results")
    
    parser.add_argument("--input", type=str, default=None,
                        help="Path to tournament results CSV (default: latest file)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Directory for analysis outputs (default: simulation_results/mcts_analysis)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to analysis configuration JSON file (default: mcts_analysis_config.json)")
    
    return parser.parse_args()

def find_latest_tournament_file():
    """Find the most recent tournament results file."""
    pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         "simulation_results", "mcts_tournament", "mcts_tournament_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No tournament result files found")
    return max(files, key=os.path.getmtime)

def main():
    """Run the MCTS analysis."""
    args = parse_args()
    
    # Find input file if not specified
    input_file = args.input
    if input_file is None:
        input_file = find_latest_tournament_file()
        print(f"Using latest tournament file: {input_file}")
    
    # Set default output directory if not specified
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "simulation_results", "mcts_analysis", f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
    # Use configuration file if specified
    config_file = args.config
    if config_file is None:
        default_config = os.path.join(os.path.dirname(__file__), "mcts_analysis_config.json")
        if os.path.exists(default_config):
            config_file = default_config
            print(f"Using configuration file: {config_file}")
        else:
            print("Using default configuration settings")
    else:
        print(f"Using configuration file: {config_file}")
    
    print(f"Starting MCTS configuration analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Run analysis
    results = main_analysis.run_analysis(input_file, output_dir, config_file)
    
    # Print top configurations
    print("\nTop MCTS Configurations:")
    for i, config in enumerate(results['top_configs']):
        print(f"{i+1}. {config['config_id']}")
        print(f"   Rollout Policy: {config['rollout_policy']}")
        print(f"   Rollout Depth: {config['rollout_depth']}")
        print(f"   Exploration Weight: {config['exploration_weight']}")
        print(f"   Composite Score: {config['composite_score']:.4f}")
        print(f"   Adjusted Win Rate (draws=0.5): {config['adjusted_win_rate']:.4f}")
        
        # Print confidence intervals if available
        if 'win_rate_ci_lower' in config and 'win_rate_ci_upper' in config:
            print(f"   Win Rate 95% CI: [{config['win_rate_ci_lower']:.4f}, {config['win_rate_ci_upper']:.4f}]")
            
        print(f"   Average Win Rate (wins only): {config['average_win_rate']:.4f}")
        print(f"   Elo Rating: {config['elo_rating']:.1f}")
        
        # Print ELO confidence intervals if available
        if 'elo_ci_lower' in config and 'elo_ci_upper' in config:
            print(f"   Elo 95% CI: [{config['elo_ci_lower']:.1f}, {config['elo_ci_upper']:.1f}]")
            
        print("")
    
    print(f"Analysis complete. Results available in: {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'mcts_analysis_summary.txt')}")

if __name__ == "__main__":
    main() 