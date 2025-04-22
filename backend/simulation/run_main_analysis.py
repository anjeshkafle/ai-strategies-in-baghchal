#!/usr/bin/env python3
"""
Run analysis on MCTS vs Minimax competition results.

This script analyzes competition results to determine the relative performance 
of MCTS and Minimax algorithms in the Bagh Chal game.

Example usage:
    # Analyze results from latest competition
    python run_main_analysis.py
    
    # Specify input file and output directory
    python run_main_analysis.py --input path/to/results.csv --output path/to/output
    
    # Use custom configuration file
    python run_main_analysis.py --config path/to/config.json
"""

import os
import sys
import argparse
import glob
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from main_analysis import main_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MCTS vs Minimax competition results")
    
    parser.add_argument("--input", type=str, default=None,
                        help="Path to competition results CSV (default: latest file)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Directory for analysis outputs (default: simulation_results/main_analysis)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to analysis configuration JSON file (default: main_analysis_config.json)")
    
    return parser.parse_args()

def find_latest_competition_file():
    """Find the most recent competition results file."""
    pattern = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         "simulation_results", "main_competition", "main_competition*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No competition result files found")
    return max(files, key=os.path.getmtime)

def main():
    """Run the competition analysis."""
    args = parse_args()
    
    # Find input file if not specified
    input_file = args.input
    if input_file is None:
        input_file = find_latest_competition_file()
        print(f"Using latest competition file: {input_file}")
    
    # Set default output directory if not specified
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "simulation_results", "main_analysis", f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
    # Use configuration file if specified
    config_file = args.config
    if config_file is None:
        default_config = os.path.join(os.path.dirname(__file__), "main_analysis_config.json")
        if os.path.exists(default_config):
            config_file = default_config
            print(f"Using configuration file: {config_file}")
        else:
            print("Using default configuration settings")
    else:
        print(f"Using configuration file: {config_file}")
    
    print(f"Starting MCTS vs Minimax competition analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Run analysis
    results = main_analysis.run_analysis(input_file, output_dir, config_file)
    
    # Print top configurations
    print("\nAlgorithm Performance:")
    algorithm_comparison = results['performance_metrics']['algorithm_comparison']
    for i, row in algorithm_comparison.iterrows():
        print(f"{row['Algorithm']}: Win Rate = {row['Win Rate']:.4f} (95% CI: [{row['CI Lower']:.4f}, {row['CI Upper']:.4f}])")
    
    print("\nTop MCTS Configurations:")
    for i, config in enumerate(results['performance_metrics']['top_mcts_configs']):
        print(f"{i+1}. {config['config_id']}")
        print(f"   Win Rate: {config['win_rate']:.4f}")
        print(f"   As Tiger: {config['tiger_win_rate']:.4f}")
        print(f"   As Goat: {config['goat_win_rate']:.4f}")
        
    print("\nTop Minimax Depths:")
    for i, config in enumerate(results['performance_metrics']['top_minimax_configs']):
        print(f"{i+1}. Depth {config['depth']}")
        print(f"   Win Rate: {config['win_rate']:.4f}")
        print(f"   As Tiger: {config['tiger_win_rate']:.4f}")
        print(f"   As Goat: {config['goat_win_rate']:.4f}")
        print(f"   Avg Move Time: {config['avg_move_time']:.2f}s")
    
    print("\nGame Dynamics Summary:")
    print(f"Average Game Length: {results['game_dynamics']['avg_game_length']:.2f} moves")
    print(f"Tiger Wins: {results['game_dynamics']['avg_length_tiger_win']:.2f} moves")
    print(f"Goat Wins: {results['game_dynamics']['avg_length_goat_win']:.2f} moves")
    print(f"Draws: {results['game_dynamics']['avg_length_draw']:.2f} moves")
    
    print(f"\nAnalysis complete. Results available in: {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'competition_analysis_summary.txt')}")

if __name__ == "__main__":
    main() 