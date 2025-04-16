"""
Utility functions for genetic algorithm tuning of MinimaxAgent.
"""
import os
import json
import time
import logging
import csv
from typing import Dict, Any, List
import matplotlib.pyplot as plt


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def ensure_output_directory(output_dir: str) -> str:
    """
    Ensure output directory exists and return its absolute path.
    
    Args:
        output_dir: Directory path (relative or absolute)
        
    Returns:
        Absolute path to the directory
    """
    if not os.path.isabs(output_dir):
        # Convert relative path to absolute based on this file's location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the genetic algorithm.
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    log_path = os.path.join(output_dir, f"ga_log_{int(time.time())}.txt")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("GeneticAlgorithm")


def log_generation_to_csv(output_dir: str, generation_data: Dict, csv_path: str = None):
    """
    Log generation data to a CSV file. Creates the file if it doesn't exist.
    Appends data to the existing file for fault tolerance.
    
    Args:
        output_dir: Directory for CSV files
        generation_data: Dictionary containing generation data
        csv_path: Optional custom path for the CSV file
    """
    if csv_path is None:
        csv_path = os.path.join(output_dir, "ga_progress.csv")
    
    # Define fields to log
    fields = [
        'generation', 'timestamp', 'best_fitness', 'avg_fitness', 
        'min_fitness', 'std_dev', 'elapsed_time', 'total_time',
        'best_tiger_win_rate', 'best_goat_win_rate', 'population_diversity'
    ]
    
    # Check if file exists to determine if header needs to be written
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row, handling missing fields
        row_data = {field: generation_data.get(field, '') for field in fields}
        writer.writerow(row_data)


def plot_fitness_history(history_file: str, output_path: str = None):
    """
    Plot fitness history from a saved results file.
    
    Args:
        history_file: Path to history JSON file
        output_path: Path to save the plot (or None to display)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Please install it with 'pip install matplotlib'")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    generations = [entry['generation'] for entry in history]
    best_fitness = [entry['best_fitness'] for entry in history]
    avg_fitness = [entry['avg_fitness'] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r-', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm Fitness History')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_parameter_evolution(csv_path: str, output_path: str = None):
    """
    Plot the evolution of parameters over generations from the CSV file.
    
    Args:
        csv_path: Path to the CSV file with parameter data
        output_path: Path to save the plot (or None to display)
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError:
        print("matplotlib and pandas are required for plotting. Please install with 'pip install matplotlib pandas'")
        return
    
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot fitness metrics
    axs[0].plot(df['generation'], df['best_fitness'], 'b-', label='Best Fitness')
    axs[0].plot(df['generation'], df['avg_fitness'], 'r-', label='Average Fitness')
    if 'min_fitness' in df.columns:
        axs[0].plot(df['generation'], df['min_fitness'], 'g-', label='Min Fitness')
    axs[0].set_ylabel('Fitness')
    axs[0].set_title('Fitness Metrics Over Generations')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot win rates
    if 'best_tiger_win_rate' in df.columns and 'best_goat_win_rate' in df.columns:
        axs[1].plot(df['generation'], df['best_tiger_win_rate'], 'orange', label='Tiger Win Rate')
        axs[1].plot(df['generation'], df['best_goat_win_rate'], 'purple', label='Goat Win Rate')
        axs[1].set_ylabel('Win Rate')
        axs[1].set_title('Win Rates for Best Chromosome')
        axs[1].legend()
        axs[1].grid(True)
    
    # Plot diversity
    if 'population_diversity' in df.columns:
        axs[2].plot(df['generation'], df['population_diversity'], 'k-')
        axs[2].set_ylabel('Diversity')
        axs[2].set_title('Population Diversity')
        axs[2].set_xlabel('Generation')
        axs[2].grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def calculate_population_diversity(population):
    """
    Calculate diversity metric for the population based on gene variance.
    
    Args:
        population: List of chromosomes
        
    Returns:
        Float representing diversity (higher means more diverse)
    """
    if not population:
        return 0
    
    # Extract all gene values into a dictionary of lists
    genes_by_param = {}
    for chromosome in population:
        for param, value in chromosome.genes.items():
            if param not in genes_by_param:
                genes_by_param[param] = []
            genes_by_param[param].append(value)
    
    # Calculate variance for each parameter
    variances = []
    for param, values in genes_by_param.items():
        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            
            # Normalize by the parameter's range if available
            if hasattr(population[0], 'config'):
                config = population[0].config
                if param in config.get("parameter_ranges", {}):
                    min_val, max_val = config["parameter_ranges"][param]
                    range_size = max_val - min_val
                    if range_size > 0:
                        variance /= (range_size ** 2)
                elif param in config.get("equilibrium_ranges", {}):
                    min_val, max_val = config["equilibrium_ranges"][param]
                    range_size = max_val - min_val
                    if range_size > 0:
                        variance /= (range_size ** 2)
            
            variances.append(variance)
    
    # Return average variance across all parameters
    if variances:
        diversity = sum(variances) / len(variances)
        return diversity
    return 0


def generate_report(output_dir: str, config: Dict[str, Any]):
    """
    Generate a report summarizing the GA optimization results.
    
    Args:
        output_dir: Directory with results
        config: GA configuration
    """
    best_params_path = os.path.join(output_dir, "best_params.json")
    if not os.path.exists(best_params_path):
        print("No best parameters found to generate report")
        return
    
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    # Get a list of generation files
    generation_files = [f for f in os.listdir(output_dir) if f.startswith("generation_") and f.endswith(".json")]
    generation_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    # Load fitness history
    fitness_history = []
    for gen_file in generation_files:
        with open(os.path.join(output_dir, gen_file), 'r') as f:
            gen_data = json.load(f)
            fitness_history.append({
                'generation': gen_data['generation'],
                'best_fitness': gen_data['best_fitness'],
                'avg_fitness': gen_data['avg_fitness']
            })
    
    # Save the fitness history
    history_path = os.path.join(output_dir, "fitness_history.json")
    with open(history_path, 'w') as f:
        json.dump(fitness_history, f, indent=2)
    
    # Generate a plot
    plot_path = os.path.join(output_dir, "fitness_plot.png")
    plot_fitness_history(history_path, plot_path)
    
    # Plot parameter evolution if CSV exists
    csv_path = os.path.join(output_dir, "ga_progress.csv")
    if os.path.exists(csv_path):
        param_plot_path = os.path.join(output_dir, "parameter_evolution.png")
        plot_parameter_evolution(csv_path, param_plot_path)
    
    # Create a report file
    report_path = os.path.join(output_dir, "optimization_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== MinimaxAgent Genetic Optimization Report ===\n\n")
        
        f.write("Configuration:\n")
        for key, value in config.items():
            if not isinstance(value, dict):
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Best Parameters:\n")
        genes = best_params.get("genes", best_params)
        for key, value in genes.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Fitness History:\n")
        for entry in fitness_history:
            f.write(f"  Generation {entry['generation']}: Best={entry['best_fitness']:.4f}, Avg={entry['avg_fitness']:.4f}\n")
        
        # Add comparison with baseline parameters
        f.write("\nComparison with Default Parameters:\n")
        f.write("  Default parameters represent the manually tuned values.\n")
        f.write("  The genetic algorithm found parameters with fitness improvement of ")
        if isinstance(best_params, dict) and "fitness" in best_params:
            f.write(f"{best_params['fitness']:.4f} over baseline.\n")
        else:
            f.write("an unknown amount over baseline.\n")
        
        # Add summary of impact
        f.write("\nParameter Impact Analysis:\n")
        f.write("  Most influential parameters based on correlation with fitness:\n")
        f.write("  (This is an estimate based on the final generation data)\n")
        
        # Include CSV statistics if available
        if os.path.exists(csv_path):
            f.write("\nTraining Statistics:\n")
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                gen_count = len(df)
                total_time = df['total_time'].max() if 'total_time' in df.columns else "Unknown"
                initial_best = df['best_fitness'].iloc[0] if not df.empty else "Unknown"
                final_best = df['best_fitness'].iloc[-1] if not df.empty else "Unknown"
                improvement = float(final_best) - float(initial_best) if isinstance(initial_best, (int, float)) and isinstance(final_best, (int, float)) else "Unknown"
                
                f.write(f"  Total Generations: {gen_count}\n")
                f.write(f"  Total Training Time: {total_time} seconds\n")
                f.write(f"  Initial Best Fitness: {initial_best}\n")
                f.write(f"  Final Best Fitness: {final_best}\n")
                f.write(f"  Overall Improvement: {improvement}\n")
            except Exception as e:
                f.write(f"  Error analyzing CSV data: {e}\n")
    
    print(f"Report generated at {report_path}")
    return report_path 