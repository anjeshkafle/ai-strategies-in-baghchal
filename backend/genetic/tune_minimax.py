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
import traceback
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the base directory for backend (parent of genetic)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set up absolute paths relative to backend
output_dir = os.path.join(base_dir, "tuned_params")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Setup minimal logging - only core functionality
logger = logging.getLogger('tune_minimax')
logger.setLevel(logging.INFO)

# Check if handlers already exist to avoid duplicate handlers
if not logger.handlers:
    # Use file handler for comprehensive logging to a single file
    file_handler = logging.FileHandler(os.path.join(output_dir, 'tuning.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Use console handler for minimal output to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

# Configure other subsystem loggers - keep only essential information
for logger_name in ['genetic_optimizer', 'fitness_evaluator']:
    sublogger = logging.getLogger(logger_name)
    sublogger.setLevel(logging.WARNING)  # Only warnings and errors
    
    # Remove any existing handlers
    for handler in sublogger.handlers[:]:
        sublogger.removeHandler(handler)
    
    # Add handlers from main logger
    for handler in logger.handlers:
        sublogger.addHandler(handler)

# Disable debug logger that was creating too many files
debug_logger = logging.getLogger('ga_debug')
debug_logger.setLevel(logging.ERROR)  # Only serious errors
debug_logger.handlers = []  # Remove all handlers
debug_logger.addHandler(logging.NullHandler())  # Add null handler

# Import modules after setting up logging
from models.minimax_agent import MinimaxAgent
from genetic.genetic_optimizer import GeneticOptimizer
from genetic.utils import (
    load_config, 
    ensure_output_directory, 
    generate_report,
    plot_fitness_history
)


def main():
    """
    Main entry point for the minimax tuning process.
    """
    try:
        logger.info("=== Starting MinimaxAgent parameter tuning ===")
        
        # Get the path to the configuration file
        config_path = os.path.join(os.path.dirname(__file__), "ga_config.json")
        logger.info(f"Loading configuration from {config_path}")
        
        # Load the configuration
        config = load_config(config_path)
        
        # Set up output directory - ALWAYS use "tuned_params" in backend directory
        global output_dir
        
        # Ensure the directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Results will be saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory {output_dir}: {e}")
            sys.exit(1)  # Exit if we can't create the directory
        
        # Setup main execution parameters
        population_size = config.get("population_size", 10)
        generations = config.get("generations", 20) 
        max_execution_time = config.get("max_execution_time", float('inf'))
        search_depth = config.get("search_depth", 3)
        
        # Print execution parameters - cleaner output
        print("\n=== Genetic Algorithm Tuning ===")
        print(f"Population: {population_size} | Generations: {generations} | Search Depth: {search_depth}")
        
        if max_execution_time < float('inf'):
            print(f"Running for up to {max_execution_time} seconds or {generations} generations")
        else:
            print(f"Running for {generations} generations")
            
        print("Press Ctrl+C to interrupt (best results so far will be saved)")
        print("...")
        
        # Create the genetic optimizer
        logger.info("Creating genetic optimizer")
        start_time = time.time()
        
        # Create the optimizer
        optimizer = GeneticOptimizer(config)
        
        # Check if we're resuming from a previous run
        population_file = os.path.join(output_dir, "population.pkl")
        generation_file = os.path.join(output_dir, "generation_count.txt")
        
        if os.path.exists(population_file) and os.path.exists(generation_file):
            with open(generation_file, 'r') as f:
                generation = int(f.read().strip())
            print(f"Resuming from generation {generation}")
        
        # Run the optimization
        try:
            logger.info("Running optimization")
            if max_execution_time < float('inf'):
                best_chromosome = optimizer.run_with_time_limit(max_execution_time)
            else:
                best_chromosome = optimizer.run()
                
            logger.info(f"Optimization completed successfully in {time.time() - start_time:.2f}s")
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user.")
            print("\nOptimization interrupted by user.")
            print("Best parameters found so far have been saved.")
            
            # Get the best chromosome found so far
            best_chromosome = optimizer.best_chromosome
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            logger.error(traceback.format_exc())
            print(f"Error during optimization: {e}")
            # Try to get the best chromosome if available
            best_chromosome = optimizer.best_chromosome if hasattr(optimizer, 'best_chromosome') else None
        
        # Check if we have a best chromosome
        if best_chromosome:
            logger.info(f"Best chromosome found with fitness: {best_chromosome.fitness}")
            
            # Generate report
            logger.info("Generating final report")
            report_path = generate_report(output_dir, config)
            
            # Generate fitness plot directly
            history_path = os.path.join(output_dir, "fitness_history.json")
            plot_path = os.path.join(output_dir, "fitness_plot.png")
            if os.path.exists(history_path):
                try:
                    logger.info("Generating fitness plot")
                    plot_fitness_history(history_path, plot_path)
                    logger.info(f"Fitness plot saved to {plot_path}")
                except Exception as e:
                    logger.error(f"Error generating fitness plot: {e}")
            
            # Final message - clean output
            print("\n=== Optimization Complete ===")
            print(f"Best fitness: {best_chromosome.fitness:.4f}")
            print(f"Results saved in: {output_dir}")
            print(f"Check {output_dir}/optimization_report.txt for a summary report.")
            print(f"Fitness plot: {plot_path}")
            print(f"Total time: {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("No best parameters found to generate report")
            print("No best parameters found to generate report")
        
        logger.info(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Unexpected error: {e}")
        sys.exit(1)


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
            
    except Exception as e:
        logger.error(f"Error testing tuned parameters: {e}")


if __name__ == "__main__":
    main() 