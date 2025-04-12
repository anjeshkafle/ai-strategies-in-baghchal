#!/usr/bin/env python3
"""
Run simulation tasks based on configuration file contents.

The script will run:
- MCTS tournament if mcts_tournament section exists
- Main competition if main_competition section exists
- Both in order if both sections exist
"""

import os
import sys
from datetime import datetime
import json
import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.simulation_controller import SimulationController
from simulation.config import load_config, save_config, SimulationConfig
from simulation.analysis import TournamentAnalyzer

def find_best_mcts_config(results_file: str) -> dict:
    """
    Analyze MCTS tournament results to find the best configuration.
    
    Args:
        results_file: Path to the MCTS tournament results CSV
        
    Returns:
        Best MCTS configuration dictionary
    """
    # Read the results
    df = pd.read_csv(results_file)
    
    # Group by configuration and calculate win rates
    config_stats = []
    
    for config_str in df['tiger_config'].unique():
        # Get games where this config was tiger
        tiger_games = df[df['tiger_config'] == config_str]
        tiger_wins = tiger_games[tiger_games['winner'] == 'tiger'].shape[0]
        tiger_win_rate = tiger_wins / len(tiger_games)
        
        # Get games where this config was goat
        goat_games = df[df['goat_config'] == config_str]
        goat_wins = goat_games[goat_games['winner'] == 'goat'].shape[0]
        goat_win_rate = goat_wins / len(goat_games)
        
        # Calculate overall win rate
        total_games = len(tiger_games) + len(goat_games)
        total_wins = tiger_wins + goat_wins
        overall_win_rate = total_wins / total_games
        
        config_stats.append({
            'config': json.loads(config_str),
            'tiger_win_rate': tiger_win_rate,
            'goat_win_rate': goat_win_rate,
            'overall_win_rate': overall_win_rate,
            'total_games': total_games
        })
    
    # Sort by overall win rate and return best config
    best_config = max(config_stats, key=lambda x: x['overall_win_rate'])
    return best_config['config']

def run_mcts_tournament(config_path: str):
    """Run the MCTS tournament using configuration file."""
    config = load_config(config_path)
    mcts_config = config.mcts_tournament
    
    print(f"Starting MCTS tournament at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    
    # Get all configurations that will be tested
    all_configs = mcts_config.get_all_configs()
    policies = set()
    iterations = set()
    depths = set()
    
    # Extract unique values for reporting
    for config in all_configs:
        policies.add(config['rollout_policy'])
        iterations.add(config['iterations'])
        depths.add(config['rollout_depth'])
    
    print(f"  Configuration groups: {len(mcts_config.configurations)}")
    print(f"  Unique policies: {sorted(policies)}")
    print(f"  Unique iterations: {sorted(iterations)}")
    print(f"  Unique depths: {sorted(depths)}")
    print(f"  Total unique configurations: {len(all_configs)}")
    print(f"  Max simulation time: {mcts_config.max_simulation_time} minutes")
    print(f"  Output directory: {mcts_config.output_dir}")
    print(f"  Parallel games: {mcts_config.parallel_games if mcts_config.parallel_games else 'auto'}")
    print()
    
    controller = SimulationController(output_dir=mcts_config.output_dir)
    
    # If parallel ranges are specified, run each range
    if mcts_config.parallel_ranges:
        for range_config in mcts_config.parallel_ranges:
            start_idx = range_config.get('start', 0)
            end_idx = range_config.get('end')
            
            print(f"Processing matchups from index {start_idx} to {end_idx if end_idx is not None else 'end'}")
            
            # Get all MCTS configs from all configuration groups
            mcts_configs = mcts_config.get_all_configs()
            
            output_file = controller.run_mcts_tournament(
                mcts_configs=mcts_configs,  # Pass pre-generated configs
                max_simulation_time=mcts_config.max_simulation_time,
                start_idx=start_idx,
                end_idx=end_idx,
                parallel_games=mcts_config.parallel_games
            )
            
            print(f"Tournament progress saved to: {output_file}")
    else:
        # Run all matchups
        # Get all MCTS configs from all configuration groups
        mcts_configs = mcts_config.get_all_configs()
        
        output_file = controller.run_mcts_tournament(
            mcts_configs=mcts_configs,  # Pass pre-generated configs
            max_simulation_time=mcts_config.max_simulation_time,
            parallel_games=mcts_config.parallel_games
        )
        
        print(f"Tournament progress saved to: {output_file}")
    
    return output_file

def run_main_competition(config_path: str):
    """Run the main competition using configuration file."""
    config = load_config(config_path)
    main_config = config.main_competition
    
    # If MCTS tournament results are specified, find the best config
    if not main_config.mcts_tournament_results:
        raise ValueError("mcts_tournament_results must be specified in the main_competition section")
    
    print(f"Analyzing MCTS tournament results from: {main_config.mcts_tournament_results}")
    
    # Use the analyzer
    analyzer = TournamentAnalyzer(main_config.output_dir)
    try:
        best_config = analyzer.find_best_config()
        print("\nBest MCTS configuration found:")
        print(f"  Policy: {best_config['config']['rollout_policy']}")
        print(f"  Iterations: {best_config['config']['iterations']}")
        print(f"  Depth: {best_config['config']['rollout_depth']}")
        print(f"  Overall Win Rate: {best_config['stats']['overall_win_rate']:.2%}")
        print(f"  95% CI: [{best_config['stats']['overall_ci'][0]:.2%}, {best_config['stats']['overall_ci'][1]:.2%}]")
        print(f"  Total Games: {best_config['stats']['total_games']}")
        
        # Print full analysis
        analyzer.print_analysis()
        
    except ValueError as e:
        raise ValueError(f"Error analyzing tournament results: {e}")
    
    print(f"Starting main competition at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Minimax depths: {main_config.minimax_depths}")
    print(f"  Games per matchup: {main_config.games_per_matchup}")
    print(f"  Output directory: {main_config.output_dir}")
    print()
    
    controller = SimulationController(output_dir=main_config.output_dir)
    
    output_file = controller.run_main_competition(
        best_mcts_config=best_config['config'],
        minimax_depths=main_config.minimax_depths,
        games_per_matchup=main_config.games_per_matchup
    )
    
    print(f"Main competition progress saved to: {output_file}")
    return output_file

def main():
    """Main entry point."""
    config_path = "simulation_config.json"
    
    # Load config (this will create a default one only if it doesn't exist)
    config = load_config(config_path)
    
    # Determine what to run based on config contents
    run_mcts = hasattr(config, 'mcts_tournament') and config.mcts_tournament is not None
    run_main = hasattr(config, 'main_competition') and config.main_competition is not None
    
    # Check if the configuration specifies any tasks
    if not run_mcts and not run_main:
        print("\nWarning: Config file doesn't contain any valid simulation tasks.")
        print("To run a simulation, add either mcts_tournament or main_competition section to your config file.")
        return
    
    # Run MCTS tournament if configured
    if run_mcts:
        print("\n=== Running MCTS Tournament ===")
        try:
            run_mcts_tournament(config_path)
            print("\n=== MCTS Tournament Completed ===")
        except Exception as e:
            print(f"\nError in MCTS Tournament: {e}")
            import traceback
            traceback.print_exc()
    
    # Run main competition if configured
    if run_main:
        print("\n=== Running Main Competition ===")
        try:
            run_main_competition(config_path)
            print("\n=== Main Competition Completed ===")
        except ValueError as e:
            print(f"\nSkipping Main Competition: {e}")
            print("Hint: To run the main competition, make sure to specify mcts_tournament_results in your config.")
        except Exception as e:
            print(f"\nError in Main Competition: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 