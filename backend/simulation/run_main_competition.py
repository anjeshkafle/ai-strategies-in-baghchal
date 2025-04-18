#!/usr/bin/env python3
"""
Main entry point for running the competition between MCTS and Minimax agents.
"""
import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.main_competition_controller import MainCompetitionController
from simulation.mcts_config import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run main competition between MCTS and Minimax')
    parser.add_argument('--config', type=str, default='main_competition_config.json',
                        help='Path to competition configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to store results (default: auto-generated)')
    parser.add_argument('--max-time', type=int, default=None,
                        help='Maximum simulation time in minutes')
    parser.add_argument('--games-per-matchup', type=int, default=None,
                        help='Number of games to play per matchup')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Number of games to run in parallel')
    parser.add_argument('--sheets-url', type=str, default=None,
                        help='Google Sheets webapp URL to sync results')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Google Sheets batch size for syncing')
    return parser.parse_args()

def find_latest_mcts_analysis():
    """Find the latest MCTS analysis directory."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'simulation_results', 'mcts_analysis')
    
    if not os.path.exists(base_dir):
        print(f"Warning: MCTS analysis directory not found at {base_dir}")
        return None
    
    # Find subdirectories with 'analysis_' prefix
    analysis_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('analysis_')]
    
    if not analysis_dirs:
        print(f"Warning: No analysis directories found in {base_dir}")
        return None
    
    # Sort by timestamp (assuming format is analysis_YYYYMMDD_HHMMSS)
    analysis_dirs.sort(reverse=True)
    latest_dir = os.path.join(base_dir, analysis_dirs[0])
    
    # Verify top_configs.csv exists
    top_configs_path = os.path.join(latest_dir, 'data', 'top_configs.csv')
    if not os.path.exists(top_configs_path):
        print(f"Warning: Top configs file not found at {top_configs_path}")
        return None
    
    return latest_dir

def main():
    """Run the main MCTS vs Minimax competition."""
    args = parse_args()
    
    # Load main competition config from the specified path or default
    main_config_path = args.config
    if not os.path.isabs(main_config_path):
        main_config_path = os.path.join(os.path.dirname(__file__), main_config_path)
    
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r') as f:
            main_config = json.load(f)
        print(f"Loaded configuration from {main_config_path}")
    else:
        print(f"Warning: Configuration file {main_config_path} not found. Using defaults.")
        main_config = {}
    
    # Find latest MCTS analysis results
    latest_analysis_dir = find_latest_mcts_analysis()
    if latest_analysis_dir:
        top_configs_path = os.path.join(latest_analysis_dir, 'data', 'top_configs.csv')
        print(f"Using top MCTS configurations from: {top_configs_path}")
    else:
        print("Error: Cannot find MCTS top configurations. Please run MCTS analysis first.")
        sys.exit(1)
    
    # Create output directory
    output_base_dir = args.output_dir or main_config.get('output_dir') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'simulation_results', 'main_competition'
    )
    
    # Ensure base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Use the base directory directly - no timestamped subfolders
    output_dir = output_base_dir
    
    # Look for existing CSV files to resume from
    existing_csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and f.startswith('main_competition')]
    
    # Sort files by timestamp (newest first)
    existing_csv_files.sort(reverse=True)
    
    # Use existing file if available
    existing_file = None
    if existing_csv_files:
        existing_file = os.path.join(output_dir, existing_csv_files[0])
        print(f"Found existing results file: {existing_file}")
        print("Will resume from this file.")
    
    # Use command line arguments if provided, otherwise use values from config
    sheets_url = args.sheets_url or main_config.get('sheets_webapp_url')
    batch_size = args.batch_size or main_config.get('sheets_batch_size', 50)
    max_simulation_time = args.max_time or main_config.get('max_simulation_time', 120)
    games_per_matchup = args.games_per_matchup  # Default to None for time-limited run
    parallel_games = args.parallel or main_config.get('parallel_games', 0)
    minimax_depths = main_config.get('minimax_depths', [3, 5, 7])
    max_time_per_move = main_config.get('max_time_per_move', 1)
    
    # Initialize controller
    controller = MainCompetitionController(
        mcts_configs_path=top_configs_path,
        minimax_depths=minimax_depths,
        output_dir=output_dir,
        sheets_url=sheets_url,
        batch_size=batch_size,
        max_time_per_move=max_time_per_move
    )
    
    # Run the competition
    print("\nStarting main competition between MCTS and Minimax agents")
    print(f"Output directory: {output_dir}")
    print(f"Minimax depths: {minimax_depths}")
    print(f"Max time per move: {max_time_per_move} seconds")
    print(f"Games per matchup: {games_per_matchup if games_per_matchup is not None else '∞ (time-limited)'}")
    print(f"Max simulation time: {max_simulation_time} minutes")
    print(f"Parallel games: {parallel_games if parallel_games > 0 else 'auto'}")
    
    result_file = controller.run_competition(
        games_per_matchup=games_per_matchup,
        max_simulation_time=max_simulation_time,
        parallel_games=parallel_games,
        output_file=existing_file
    )
    
    print(f"\nMain competition complete! Results saved to: {result_file}")
    return result_file

if __name__ == "__main__":
    main() 