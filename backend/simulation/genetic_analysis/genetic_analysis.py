"""
Main script for genetic algorithm tuning analysis.
"""
import os
import json
import pandas as pd
from . import visualization
from . import utils

def run_analysis(source_dir, output_dir, config_file=None):
    """
    Run complete analysis pipeline on genetic algorithm tuning results.
    
    Args:
        source_dir: Directory containing genetic algorithm tuning results
        output_dir: Directory for output files
        config_file: Path to configuration JSON file
    
    Returns:
        Dictionary with analysis results
    """
    # Load configuration
    config = utils.load_config(config_file)
    
    # Create output directories
    data_dir = os.path.join(output_dir, 'data')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading genetic algorithm data from {source_dir}...")
    
    # Load and process data
    ga_data = utils.load_and_preprocess_data(source_dir)
    
    print(f"Analyzing genetic algorithm tuning across {ga_data['total_generations']} generations...")
    
    # Perform fitness analysis
    fitness_analysis = analyze_fitness(ga_data, config)
    print("Fitness analysis completed")
    
    # Perform parameter evolution analysis
    parameter_analysis = analyze_parameters(ga_data, config)
    print("Parameter analysis completed")
    
    # Perform win rate analysis
    win_rate_analysis = analyze_win_rates(ga_data, config)
    print("Win rate analysis completed")
    
    # Perform diversity analysis
    diversity_analysis = analyze_diversity(ga_data, config)
    print("Diversity analysis completed")
    
    # Perform time efficiency analysis
    time_analysis = analyze_time_efficiency(ga_data, config)
    print("Time efficiency analysis completed")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Only generate visualizations enabled in the config
    if config.get('analysis', {}).get('include_fitness_analysis', True):
        visualization.create_fitness_visualizations(fitness_analysis, figures_dir, config)
    
    if config.get('analysis', {}).get('include_parameter_evolution', True):
        visualization.create_parameter_evolution_visualizations(parameter_analysis, figures_dir, config)
    
    if config.get('analysis', {}).get('include_win_rate_analysis', True):
        visualization.create_win_rate_visualizations(win_rate_analysis, figures_dir, config)
    
    if config.get('analysis', {}).get('include_diversity_analysis', True):
        visualization.create_diversity_visualizations(diversity_analysis, figures_dir, config)
    
    if config.get('analysis', {}).get('include_time_efficiency', True):
        visualization.create_time_efficiency_visualizations(time_analysis, figures_dir, config)
    
    if config.get('analysis', {}).get('include_parameter_comparison', True):
        visualization.create_parameter_comparison_visualizations(parameter_analysis, figures_dir, config)
    
    # Save processed data
    print("\nSaving processed data...")
    
    # Save fitness data
    fitness_df = pd.DataFrame(fitness_analysis['fitness_history'])
    fitness_df.to_csv(os.path.join(data_dir, "fitness_history.csv"), index=False)
    
    # Save parameter evolution data
    parameter_df = pd.DataFrame(parameter_analysis['parameter_evolution'])
    parameter_df.to_csv(os.path.join(data_dir, "parameter_evolution.csv"), index=False)
    
    # Save win rate data
    if 'win_rate_history' in win_rate_analysis:
        win_rate_df = pd.DataFrame(win_rate_analysis['win_rate_history'])
        win_rate_df.to_csv(os.path.join(data_dir, "win_rate_history.csv"), index=False)
    
    # Save diversity data
    if 'diversity_history' in diversity_analysis:
        diversity_df = pd.DataFrame(diversity_analysis['diversity_history'])
        diversity_df.to_csv(os.path.join(data_dir, "diversity_history.csv"), index=False)
    
    # Save time efficiency data
    if 'time_history' in time_analysis:
        time_df = pd.DataFrame(time_analysis['time_history'])
        time_df.to_csv(os.path.join(data_dir, "time_history.csv"), index=False)
    
    # Generate a summary report
    generate_summary_report(
        output_dir, ga_data, fitness_analysis, parameter_analysis, 
        win_rate_analysis, diversity_analysis, time_analysis
    )
    
    # Prepare results to return
    results = {
        'optimization_summary': {
            'total_generations': ga_data['total_generations'],
            'initial_best_fitness': fitness_analysis['initial_best_fitness'],
            'final_best_fitness': fitness_analysis['final_best_fitness'],
            'overall_improvement': fitness_analysis['overall_improvement'],
        },
        'parameter_analysis': {
            'top_parameters': parameter_analysis['top_parameter_changes'],
        },
        'time_analysis': {
            'total_time': time_analysis['total_time'],
            'avg_time_per_generation': time_analysis['avg_time_per_generation'],
        },
    }
    
    # Add win rates if available
    if ('best_tiger_win_rate' in win_rate_analysis and 
        'best_goat_win_rate' in win_rate_analysis):
        results['win_rates'] = {
            'tiger_win_rate': win_rate_analysis['best_tiger_win_rate'],
            'goat_win_rate': win_rate_analysis['best_goat_win_rate'],
        }
        
        # Add adjusted win rates if available
        if ('best_tiger_adjusted_win_rate' in win_rate_analysis and
            'best_goat_adjusted_win_rate' in win_rate_analysis):
            results['win_rates']['tiger_adjusted_win_rate'] = win_rate_analysis['best_tiger_adjusted_win_rate']
            results['win_rates']['goat_adjusted_win_rate'] = win_rate_analysis['best_goat_adjusted_win_rate']
    
    return results

def analyze_fitness(ga_data, config):
    """
    Analyze fitness trends across generations.
    
    Args:
        ga_data: Dictionary with genetic algorithm data
        config: Analysis configuration
        
    Returns:
        Dictionary with fitness analysis results
    """
    fitness_history = ga_data['fitness_history']
    
    if not fitness_history:
        return {
            'initial_best_fitness': 0,
            'final_best_fitness': 0,
            'overall_improvement': 0,
            'fitness_history': []
        }
    
    # Extract key metrics
    initial_best_fitness = fitness_history[0]['best_fitness']
    final_best_fitness = fitness_history[-1]['best_fitness']
    overall_improvement = final_best_fitness - initial_best_fitness
    
    # Calculate convergence metrics - FIX: Improve detection algorithm
    converged_at = None
    convergence_threshold = config.get('analysis', {}).get('convergence_threshold', 0.001)
    stability_count = config.get('analysis', {}).get('stability_count', 3)
    # FIX: Require at least 5 generations before declaring convergence
    min_generations_for_convergence = 5
    
    # Only check for convergence if we have enough generations
    if len(fitness_history) >= min_generations_for_convergence:
        for i in range(min_generations_for_convergence, len(fitness_history) - stability_count + 1):
            # Check if the fitness is stable for stability_count consecutive generations
            stable = True
            base_fitness = fitness_history[i-1]['best_fitness']
            
            for j in range(i, i + stability_count):
                if abs(fitness_history[j]['best_fitness'] - base_fitness) >= convergence_threshold:
                    stable = False
                    break
            
            if stable:
                converged_at = i
                break
    
    # Calculate improvement rate
    if len(fitness_history) > 1:
        total_gens = len(fitness_history)
        improvement_rate = overall_improvement / (total_gens - 1) if total_gens > 1 else 0
    else:
        improvement_rate = 0
    
    return {
        'initial_best_fitness': initial_best_fitness,
        'final_best_fitness': final_best_fitness,
        'overall_improvement': overall_improvement,
        'fitness_history': fitness_history,
        'converged_at': converged_at,
        'improvement_rate': improvement_rate
    }

def analyze_parameters(ga_data, config):
    """
    Analyze parameter evolution across generations.
    
    Args:
        ga_data: Dictionary with genetic algorithm data
        config: Analysis configuration
        
    Returns:
        Dictionary with parameter analysis results
    """
    if not ga_data.get('parameter_evolution'):
        return {
            'parameter_evolution': [],
            'top_parameter_changes': {}
        }
    
    parameter_evolution = ga_data['parameter_evolution']
    
    # Calculate parameter changes (initial vs final)
    initial_params = parameter_evolution[0]
    final_params = parameter_evolution[-1]
    
    param_changes = {}
    for param in initial_params:
        if param == 'generation' or param == 'timestamp':
            continue
        
        initial_value = initial_params[param]
        final_value = final_params[param]
        
        # Avoid division by zero
        percent_change = ((final_value - initial_value) / initial_value * 100) if initial_value != 0 else float('inf')
        
        param_changes[param] = {
            'initial_value': initial_value,
            'final_value': final_value,
            'absolute_change': final_value - initial_value,
            'percent_change': percent_change
        }
    
    # Sort by absolute percent change to find top changing parameters
    sorted_changes = sorted(param_changes.items(), key=lambda x: abs(x[1]['percent_change']), reverse=True)
    top_param_changes = dict(sorted_changes[:5]) if len(sorted_changes) > 5 else dict(sorted_changes)
    
    return {
        'parameter_evolution': parameter_evolution,
        'parameter_changes': param_changes,
        'top_parameter_changes': top_param_changes
    }

def analyze_win_rates(ga_data, config):
    """
    Analyze win rates across generations.
    
    Args:
        ga_data: Dictionary with genetic algorithm data
        config: Analysis configuration
        
    Returns:
        Dictionary with win rate analysis results
    """
    if 'win_rate_history' not in ga_data or not ga_data['win_rate_history']:
        return {
            'win_rate_history': [],
            'best_tiger_win_rate': 0,
            'best_goat_win_rate': 0
        }
    
    win_rate_history = ga_data['win_rate_history']
    
    # Process win rate history to add draw-adjusted rates
    for entry in win_rate_history:
        tiger_rate = entry.get('tiger_win_rate', 0)
        goat_rate = entry.get('goat_win_rate', 0)
        draw_rate = 1.0 - tiger_rate - goat_rate
        
        # Add draw-adjusted win rates (where draws count as 0.5 for both sides)
        entry['tiger_adjusted_win_rate'] = tiger_rate + 0.5 * draw_rate
        entry['goat_adjusted_win_rate'] = goat_rate + 0.5 * draw_rate
        entry['draw_rate'] = draw_rate
    
    # Find best win rates (raw, not adjusted)
    best_tiger_win_rate = max([entry.get('tiger_win_rate', 0) for entry in win_rate_history], default=0)
    best_goat_win_rate = max([entry.get('goat_win_rate', 0) for entry in win_rate_history], default=0)
    
    # Find best adjusted win rates
    best_tiger_adjusted = max([entry.get('tiger_adjusted_win_rate', 0) for entry in win_rate_history], default=0)
    best_goat_adjusted = max([entry.get('goat_adjusted_win_rate', 0) for entry in win_rate_history], default=0)
    
    # Calculate average win rates
    avg_tiger_win_rate = sum([entry.get('tiger_win_rate', 0) for entry in win_rate_history]) / len(win_rate_history) if win_rate_history else 0
    avg_goat_win_rate = sum([entry.get('goat_win_rate', 0) for entry in win_rate_history]) / len(win_rate_history) if win_rate_history else 0
    avg_draw_rate = sum([entry.get('draw_rate', 0) for entry in win_rate_history]) / len(win_rate_history) if win_rate_history else 0
    
    # Calculate average adjusted win rates
    avg_tiger_adjusted = avg_tiger_win_rate + 0.5 * avg_draw_rate
    avg_goat_adjusted = avg_goat_win_rate + 0.5 * avg_draw_rate
    
    return {
        'win_rate_history': win_rate_history,
        'best_tiger_win_rate': best_tiger_win_rate,
        'best_goat_win_rate': best_goat_win_rate,
        'best_tiger_adjusted_win_rate': best_tiger_adjusted,
        'best_goat_adjusted_win_rate': best_goat_adjusted,
        'avg_tiger_win_rate': avg_tiger_win_rate,
        'avg_goat_win_rate': avg_goat_win_rate,
        'avg_draw_rate': avg_draw_rate,
        'avg_tiger_adjusted_win_rate': avg_tiger_adjusted,
        'avg_goat_adjusted_win_rate': avg_goat_adjusted
    }

def analyze_diversity(ga_data, config):
    """
    Analyze population diversity across generations.
    
    Args:
        ga_data: Dictionary with genetic algorithm data
        config: Analysis configuration
        
    Returns:
        Dictionary with diversity analysis results
    """
    if 'diversity_history' not in ga_data or not ga_data['diversity_history']:
        return {
            'diversity_history': [],
            'avg_diversity': 0,
            'diversity_trend': 'unchanged'
        }
    
    diversity_history = ga_data['diversity_history']
    
    # Calculate average diversity
    avg_diversity = sum([entry.get('population_diversity', 0) for entry in diversity_history]) / len(diversity_history) if diversity_history else 0
    
    # Analyze diversity trend
    if len(diversity_history) > 1:
        initial_diversity = diversity_history[0].get('population_diversity', 0)
        final_diversity = diversity_history[-1].get('population_diversity', 0)
        
        if final_diversity > initial_diversity * 1.1:
            diversity_trend = 'increasing'
        elif final_diversity < initial_diversity * 0.9:
            diversity_trend = 'decreasing'
        else:
            diversity_trend = 'stable'
    else:
        diversity_trend = 'unchanged'
    
    return {
        'diversity_history': diversity_history,
        'avg_diversity': avg_diversity,
        'diversity_trend': diversity_trend
    }

def analyze_time_efficiency(ga_data, config):
    """
    Analyze time efficiency of the genetic algorithm.
    
    Args:
        ga_data: Dictionary with genetic algorithm data
        config: Analysis configuration
        
    Returns:
        Dictionary with time efficiency analysis results
    """
    if 'time_history' not in ga_data or not ga_data['time_history']:
        return {
            'time_history': [],
            'total_time': 0,
            'avg_time_per_generation': 0
        }
    
    time_history = ga_data['time_history']
    
    # Calculate total time and average time per generation
    total_time = sum([entry.get('elapsed_time', 0) for entry in time_history])
    avg_time_per_generation = total_time / len(time_history) if time_history else 0
    
    return {
        'time_history': time_history,
        'total_time': total_time,
        'avg_time_per_generation': avg_time_per_generation
    }

def generate_summary_report(output_dir, ga_data, fitness_analysis, parameter_analysis, 
                           win_rate_analysis, diversity_analysis, time_analysis):
    """
    Generate a summary report of the genetic algorithm analysis.
    
    Args:
        output_dir: Directory to save the report
        ga_data: Dictionary with genetic algorithm data
        fitness_analysis: Results of fitness analysis
        parameter_analysis: Results of parameter analysis
        win_rate_analysis: Results of win rate analysis
        diversity_analysis: Results of diversity analysis
        time_analysis: Results of time efficiency analysis
    """
    report_path = os.path.join(output_dir, "genetic_optimization_report.txt")
    
    with open(report_path, "w") as f:
        f.write("MinimaxAgent Genetic Algorithm Tuning Analysis\n")
        f.write("============================================\n\n")
        
        f.write("Optimization Summary\n")
        f.write("------------------\n")
        f.write(f"Total generations: {ga_data.get('total_generations', 0)}\n")
        f.write(f"Initial best fitness: {fitness_analysis.get('initial_best_fitness', 0):.4f}\n")
        f.write(f"Final best fitness: {fitness_analysis.get('final_best_fitness', 0):.4f}\n")
        f.write(f"Overall improvement: {fitness_analysis.get('overall_improvement', 0):.4f}\n")
        
        if fitness_analysis.get('converged_at') is not None:
            f.write(f"Converged at generation: {fitness_analysis['converged_at']}\n")
        
        f.write(f"Improvement rate per generation: {fitness_analysis.get('improvement_rate', 0):.6f}\n\n")
        
        f.write("Win Rate Analysis\n")
        f.write("----------------\n")
        # Raw win rates
        f.write("Raw Win Rates (Actual game outcomes):\n")
        f.write(f"Best tiger win rate: {win_rate_analysis.get('best_tiger_win_rate', 0):.4f}\n")
        f.write(f"Best goat win rate: {win_rate_analysis.get('best_goat_win_rate', 0):.4f}\n")
        f.write(f"Average tiger win rate: {win_rate_analysis.get('avg_tiger_win_rate', 0):.4f}\n")
        f.write(f"Average goat win rate: {win_rate_analysis.get('avg_goat_win_rate', 0):.4f}\n")
        f.write(f"Average draw rate: {win_rate_analysis.get('avg_draw_rate', 0):.4f}\n\n")
        
        # Adjusted win rates
        f.write("Adjusted Win Rates (Draws counted as 0.5 for each side):\n")
        f.write(f"Best tiger adjusted rate: {win_rate_analysis.get('best_tiger_adjusted_win_rate', 0):.4f}\n")
        f.write(f"Best goat adjusted rate: {win_rate_analysis.get('best_goat_adjusted_win_rate', 0):.4f}\n")
        f.write(f"Average tiger adjusted rate: {win_rate_analysis.get('avg_tiger_adjusted_win_rate', 0):.4f}\n")
        f.write(f"Average goat adjusted rate: {win_rate_analysis.get('avg_goat_adjusted_win_rate', 0):.4f}\n\n")
        
        f.write("Time Efficiency\n")
        f.write("--------------\n")
        f.write(f"Total optimization time: {time_analysis.get('total_time', 0):.2f} seconds\n")
        f.write(f"Average time per generation: {time_analysis.get('avg_time_per_generation', 0):.2f} seconds\n\n")
        
        f.write("Population Diversity\n")
        f.write("------------------\n")
        f.write(f"Average diversity: {diversity_analysis.get('avg_diversity', 0):.4f}\n")
        f.write(f"Diversity trend: {diversity_analysis.get('diversity_trend', 'unknown')}\n\n")
        
        f.write("Top Parameter Changes\n")
        f.write("-------------------\n")
        for param, details in parameter_analysis.get('top_parameter_changes', {}).items():
            f.write(f"{param}:\n")
            f.write(f"  Initial value: {details['initial_value']:.4f}\n")
            f.write(f"  Final value: {details['final_value']:.4f}\n")
            f.write(f"  Change: {details['absolute_change']:.4f} ({details['percent_change']:.2f}%)\n")
        f.write("\n")
        
        f.write("\nBest Configuration\n")
        f.write("-----------------\n")
        if ga_data.get('best_params'):
            for param, value in ga_data['best_params'].items():
                f.write(f"{param}: {value}\n")
        else:
            f.write("Best configuration not available\n")
        
    print(f"Summary report generated: {report_path}") 