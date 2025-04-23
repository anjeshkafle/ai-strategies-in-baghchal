#!/usr/bin/env python3
"""
Run analysis on genetic algorithm tuning results.

This script analyzes the results of genetic algorithm tuning for MinimaxAgent parameters
to provide insights on the tuning process, parameter evolution, and optimization effectiveness.

Example usage:
    # Analyze results from the tuned_params directory
    python run_genetic_analysis.py
    
    # Specify source directory and output directory
    python run_genetic_analysis.py --source path/to/tuned_params --output path/to/output
    
    # Use custom configuration file
    python run_genetic_analysis.py --config path/to/config.json
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from genetic_analysis import genetic_analysis

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze genetic algorithm tuning results")
    
    parser.add_argument("--source", type=str, default=None,
                        help="Path to source directory with genetic tuning results (default: backend/tuned_params)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Directory for analysis outputs (default: simulation_results/genetic_analysis)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to analysis configuration JSON file (default: genetic_analysis_config.json)")
    
    return parser.parse_args()

def main():
    """Run the genetic algorithm analysis."""
    args = parse_args()
    
    # Set default source directory if not specified
    source_dir = args.source
    if source_dir is None:
        source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tuned_params")
        print(f"Using default source directory: {source_dir}")
    
    # Set default output directory if not specified
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "simulation_results", "genetic_analysis", f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
    # Use configuration file if specified
    config_file = args.config
    if config_file is None:
        default_config = os.path.join(os.path.dirname(__file__), "genetic_analysis_config.json")
        if os.path.exists(default_config):
            config_file = default_config
            print(f"Using configuration file: {config_file}")
        else:
            print("Using default configuration settings")
    else:
        print(f"Using configuration file: {config_file}")
    
    print(f"Starting genetic algorithm analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Run analysis
    results = genetic_analysis.run_analysis(source_dir, output_dir, config_file)
    
    # Print summary of results
    print("\nGenetic Algorithm Optimization Results:")
    print(f"Total generations: {results['optimization_summary']['total_generations']}")
    print(f"Initial best fitness: {results['optimization_summary']['initial_best_fitness']:.4f}")
    print(f"Final best fitness: {results['optimization_summary']['final_best_fitness']:.4f}")
    print(f"Overall improvement: {results['optimization_summary']['overall_improvement']:.4f}")
    
    print("\nBest Parameters Performance:")
    if 'win_rates' in results:
        # Raw win rates
        print("Raw Win Rates (actual game outcomes):")
        print(f"  Tiger win rate: {results['win_rates']['tiger_win_rate']:.4f}")
        print(f"  Goat win rate: {results['win_rates']['goat_win_rate']:.4f}")
        
        # If adjusted rates are available, show them too
        if 'tiger_adjusted_win_rate' in results['win_rates']:
            print("\nAdjusted Win Rates (draws = 0.5):")
            print(f"  Tiger adjusted rate: {results['win_rates']['tiger_adjusted_win_rate']:.4f}")
            print(f"  Goat adjusted rate: {results['win_rates']['goat_adjusted_win_rate']:.4f}")
    
    print("\nTime Analysis:")
    print(f"Total optimization time: {results['time_analysis']['total_time']:.2f} seconds")
    print(f"Average time per generation: {results['time_analysis']['avg_time_per_generation']:.2f} seconds")
    
    print("\nKey Parameter Changes:")
    for param, change in results['parameter_analysis']['top_parameters'].items():
        print(f"{param}: {change['percent_change']:.2f}% change, final value: {change['final_value']:.4f}")
    
    print(f"\nAnalysis complete. Results available in: {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'genetic_optimization_report.txt')}")

if __name__ == "__main__":
    main() 