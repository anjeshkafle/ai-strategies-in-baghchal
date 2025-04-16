#!/usr/bin/env python3
"""
Genetic algorithm tuning for MinimaxAgent parameters.
Main entry point for the tuning process.

This script is the entry point for running genetic algorithm optimization of the
MinimaxAgent heuristic parameters. It handles configuration loading, initialization
of the genetic algorithm, running the optimization process, and saving results.

Usage:
    python tune_minimax.py

All configuration is read from ga_config.json. No command-line arguments are needed.
Results are saved to the output directory specified in the configuration.
"""
import os
import sys
import json
import logging
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from genetic.genetic_optimizer import GeneticOptimizer
from genetic.utils import (
    load_config, 
    ensure_output_directory, 
    setup_logging,
    generate_report
)
from genetic.params_manager import save_tuned_parameters


def main():
    """
    Main entry point for MinimaxAgent genetic tuning.
    
    This sets up the optimization process based on the configuration file,
    runs the genetic algorithm, and generates reports.
    """
    start_time = time.time()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend directory
    
    # Path to configuration file
    config_path = os.path.join(script_dir, "ga_config.json")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Set up output directory - ALWAYS use "tuned_params", ignore any config setting
    output_dir = "tuned_params"  # Hard-coded to always use "tuned_params", completely ignoring config
    
    # Make sure output_dir is resolved consistently relative to backend directory
    # It's a relative path, resolve it relative to the backend directory
    output_dir = os.path.normpath(os.path.join(base_dir, output_dir))
    
    # Ensure the directory exists - create with robust error handling
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        sys.exit(1)  # Exit if we can't create the directory
    
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting MinimaxAgent tuning with configuration from {config_path}")
    
    # Log configuration
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create and run the genetic optimizer
    print("Initializing genetic optimizer...")
    optimizer = GeneticOptimizer(config)
    
    # Get execution time limit if specified
    max_execution_time = config.get("max_execution_time", None)
    if max_execution_time:
        print(f"Optimization will run for up to {max_execution_time} seconds or "
              f"{config.get('generations', 20)} generations, whichever comes first.")
    else:
        print(f"Starting optimization with {config.get('population_size', 30)} chromosomes "
              f"for {config.get('generations', 20)} generations...")
    
    print("This may take a while. Progress is being logged to the output directory.")
    print("Press Ctrl+C to interrupt (best results so far will be saved).")
    
    try:
        # Run the optimization with time limit if specified
        if max_execution_time:
            best_chromosome = optimizer.run_with_time_limit(max_execution_time)
        else:
            best_chromosome = optimizer.run()
        
        # Save the best parameters
        best_path = os.path.join(output_dir, "best_params.json")
        best_chromosome.save_to_file(best_path)
        logger.info(f"Best parameters saved to {best_path}")
        
        # Save summary in a more accessible format
        summary_path = os.path.join(output_dir, "parameters_summary.json")
        with open(summary_path, 'w') as f:
            summary = {
                "fitness": best_chromosome.fitness,
                "parameters": best_chromosome.genes
            }
            json.dump(summary, f, indent=2)
        logger.info(f"Parameters summary saved to {summary_path}")
        
        # Generate a report
        logger.info("Generating final report...")
        report_path = generate_report(output_dir, config)
        logger.info(f"Generated report at {report_path}")
        
        # Log total time
        total_time = time.time() - start_time
        logger.info(f"Optimization completed in {total_time:.2f} seconds")
        
        # Test the tuned parameters
        logger.info("Testing tuned parameters with a sample game...")
        test_tuned_parameters(best_chromosome.genes, logger, config)
        
        print(f"\nOptimization complete! Best fitness: {best_chromosome.fitness:.4f}")
        print(f"Results saved in: {output_dir}")
        print(f"Check {report_path} for a summary report.")
        
    except KeyboardInterrupt:
        # Handle graceful interruption
        print("\nOptimization interrupted by user.")
        logger.warning("Optimization interrupted by user.")
        print("Best parameters found so far have been saved.")
        
        # Generate report with partial data
        try:
            report_path = generate_report(output_dir, config)
            print(f"Partial report generated at {report_path}")
        except Exception as e:
            logger.error(f"Error generating report after interruption: {e}")
        
        # Calculate duration
        total_time = time.time() - start_time
        logger.info(f"Optimization interrupted after {total_time:.2f} seconds")


def test_tuned_parameters(params, logger, config=None):
    """
    Test tuned parameters with a sample game.
    
    Args:
        params: Parameter dictionary
        logger: Logger instance
        config: Configuration dictionary (optional)
    """
    try:
        # Get search depth from config or use default
        search_depth = 5
        if config and "search_depth" in config:
            search_depth = config.get("search_depth")
            
        # Create agents with tuned and default parameters
        tuned_agent = MinimaxAgent(max_depth=search_depth)
        from genetic.params_manager import apply_tuned_parameters
        apply_tuned_parameters(tuned_agent, params)
        
        default_agent = MinimaxAgent(max_depth=search_depth)
        
        # Log the search depth being used
        logger.info(f"Testing with search depth: {search_depth}")
        
        # Compare the evaluation function outputs
        from models.game_state import GameState
        test_state = GameState()
        
        # Apply a few random moves to get a non-starting position
        for _ in range(5):
            moves = test_state.get_valid_moves()
            if moves:
                test_state.apply_move(moves[0])
        
        # Get evaluations
        tuned_eval = tuned_agent.evaluate(test_state)
        default_eval = default_agent.evaluate(test_state)
        
        logger.info(f"Sample evaluation - Tuned: {tuned_eval}, Default: {default_eval}")
        logger.info("Tuned parameters testing complete")
        
        # Log parameter comparison
        logger.info("\nParameter comparison - Default vs Tuned:")
        for param in [
            'mobility_weight_placement',
            'mobility_weight_movement',
            'base_capture_value',
            'capture_speed_weight',
            'dispersion_weight',
            'edge_weight',
            'closed_spaces_weight'
        ]:
            default_value = getattr(default_agent, param)
            tuned_value = getattr(tuned_agent, param)
            diff_pct = ((tuned_value - default_value) / default_value) * 100 if default_value != 0 else float('inf')
            logger.info(f"  {param}: {default_value} -> {tuned_value} ({diff_pct:.1f}% change)")
            
    except Exception as e:
        logger.error(f"Error testing tuned parameters: {e}")


if __name__ == "__main__":
    main() 