"""
Visualization functions for genetic algorithm analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns

def setup_visualization_style(config):
    """
    Set up matplotlib visualization style based on configuration.
    
    Args:
        config: Dictionary with visualization configuration
    """
    # Set style
    style = config.get('visualization', {}).get('style', 'seaborn-v0_8-whitegrid')
    plt.style.use(style)
    
    # Set DPI for high-quality output
    mpl.rcParams['figure.dpi'] = config.get('visualization', {}).get('dpi', 300)
    
    # Other common settings
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10

def create_fitness_visualizations(fitness_data, output_dir, config):
    """
    Create visualizations for fitness data.
    
    Args:
        fitness_data: Dictionary with fitness analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'fitness_history' not in fitness_data or not fitness_data['fitness_history']:
        print("No fitness data available for visualization")
        return
    
    fitness_history = fitness_data['fitness_history']
    generations = [entry['generation'] for entry in fitness_history]
    best_fitness = [entry['best_fitness'] for entry in fitness_history]
    avg_fitness = [entry.get('avg_fitness', 0) for entry in fitness_history]
    
    # Create fitness trend plot
    figsize = config.get('visualization', {}).get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    if any(avg_fitness):
        ax.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Average Fitness')
    
    # Add convergence point if available
    if fitness_data.get('converged_at') is not None:
        converged_gen = fitness_data['converged_at']
        converged_fitness = best_fitness[converged_gen-1] if converged_gen <= len(best_fitness) else best_fitness[-1]
        ax.axvline(x=converged_gen, color='g', linestyle='--', alpha=0.7, 
                 label=f'Convergence (Gen {converged_gen})')
        ax.scatter([converged_gen], [converged_fitness], color='g', s=100, zorder=5)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Evolution Over Generations')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations for initial and final fitness
    if len(best_fitness) > 1:
        ax.annotate(f'Initial: {best_fitness[0]:.4f}', 
                  xy=(generations[0], best_fitness[0]),
                  xytext=(generations[0] + 1, best_fitness[0] + 0.02),
                  arrowprops=dict(arrowstyle='->'))
        
        ax.annotate(f'Final: {best_fitness[-1]:.4f}', 
                  xy=(generations[-1], best_fitness[-1]),
                  xytext=(generations[-1] - 3, best_fitness[-1] + 0.02),
                  arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_evolution.png'))
    plt.close()
    
    # Create fitness improvement plot
    if len(best_fitness) > 1:
        improvement = [0]
        for i in range(1, len(best_fitness)):
            improvement.append(best_fitness[i] - best_fitness[i-1])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(generations, improvement, alpha=0.7, color='green')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Improvement')
        ax.set_title('Fitness Improvement Per Generation')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fitness_improvement.png'))
        plt.close()

def create_parameter_evolution_visualizations(parameter_data, output_dir, config):
    """
    Create visualizations for parameter evolution data.
    
    Args:
        parameter_data: Dictionary with parameter analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'parameter_evolution' not in parameter_data or not parameter_data['parameter_evolution']:
        print("No parameter evolution data available for visualization")
        return
    
    parameter_evolution = parameter_data['parameter_evolution']
    generations = [entry['generation'] for entry in parameter_evolution]
    
    # Get list of parameters (excluding metadata fields)
    first_entry = parameter_evolution[0]
    parameters = [param for param in first_entry.keys() 
                if param not in ['generation', 'timestamp', 'fitness']]
    
    # Create visualization for all parameters
    max_params_per_plot = 5
    chunks = [parameters[i:i + max_params_per_plot] 
            for i in range(0, len(parameters), max_params_per_plot)]
    
    for i, param_group in enumerate(chunks):
        figsize = config.get('visualization', {}).get('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        for param in param_group:
            values = [entry[param] for entry in parameter_evolution]
            ax.plot(generations, values, marker='o', markersize=4, label=param)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Parameter Value')
        ax.set_title(f'Parameter Evolution (Group {i+1})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_evolution_group{i+1}.png'))
        plt.close()
    
    # Create parameter changes visualization (initial vs final)
    if 'parameter_changes' in parameter_data and parameter_data['parameter_changes']:
        param_changes = parameter_data['parameter_changes']
        
        # Select top 10 parameters by absolute percent change
        sorted_changes = sorted(param_changes.items(), 
                             key=lambda x: abs(x[1]['percent_change']), 
                             reverse=True)
        top_params = dict(sorted_changes[:10])
        
        param_names = list(top_params.keys())
        percent_changes = [top_params[param]['percent_change'] for param in param_names]
        
        # Create bar chart for percent changes
        fig, ax = plt.subplots(figsize=figsize)
        colors = ['green' if x >= 0 else 'red' for x in percent_changes]
        bars = ax.bar(param_names, percent_changes, color=colors, alpha=0.7)
        
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Percent Change (%)')
        ax.set_title('Top Parameter Changes (Initial vs Final)')
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add labels on bars
        for bar in bars:
            height = bar.get_height()
            sign = '+' if height >= 0 else ''
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                  f'{sign}{height:.1f}%', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_percent_changes.png'))
        plt.close()

def create_parameter_sensitivity_visualizations(parameter_data, output_dir, config):
    """
    Create visualizations for parameter sensitivity analysis.
    
    Args:
        parameter_data: Dictionary with parameter analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'parameter_importance' not in parameter_data or not parameter_data['parameter_importance']:
        print("No parameter importance data available for visualization")
        return
    
    # Create parameter importance visualization
    param_importance = parameter_data['parameter_importance']
    
    # Select top 10 parameters by absolute correlation
    top_params = param_importance[:10] if len(param_importance) > 10 else param_importance
    
    param_names = [param['parameter'] for param in top_params]
    correlations = [param['correlation'] for param in top_params]
    
    # Create horizontal bar chart for correlations
    figsize = config.get('visualization', {}).get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['green' if x >= 0 else 'red' for x in correlations]
    bars = ax.barh(param_names, correlations, color=colors, alpha=0.7)
    
    ax.set_xlabel('Correlation with Fitness')
    ax.set_ylabel('Parameter')
    ax.set_title('Parameter Importance (Correlation with Fitness)')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels on bars
    for bar in bars:
        width = bar.get_width()
        sign = '+' if width >= 0 else ''
        ax.text(width + (0.01 if width >= 0 else -0.05), 
              bar.get_y() + bar.get_height()/2.,
              f'{sign}{width:.3f}', 
              ha='left' if width >= 0 else 'right', 
              va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_importance.png'))
    plt.close()

def create_parameter_comparison_visualizations(parameter_data, output_dir, config):
    """
    Create visualizations comparing initial and final parameter values.
    
    Args:
        parameter_data: Dictionary with parameter analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if ('parameter_changes' not in parameter_data or 
        not parameter_data['parameter_changes']):
        print("No parameter changes data available for visualization")
        return
    
    param_changes = parameter_data['parameter_changes']
    
    # Select a manageable number of parameters to display
    sorted_changes = sorted(param_changes.items(), 
                         key=lambda x: abs(x[1]['percent_change']), 
                         reverse=True)
    top_params = dict(sorted_changes[:10])
    
    param_names = list(top_params.keys())
    initial_values = [top_params[param]['initial_value'] for param in param_names]
    final_values = [top_params[param]['final_value'] for param in param_names]
    
    # Create bar chart comparing initial and final values
    figsize = config.get('visualization', {}).get('figsize', (12, 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(param_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, initial_values, width, label='Initial', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, final_values, width, label='Final', alpha=0.7, color='orange')
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title('Initial vs Final Parameter Values')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_initial_vs_final.png'))
    plt.close()
    
    # FIX: Create additional chart with logarithmic scale for better visibility of small values
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale to better visualize parameters with different magnitudes
    bars1 = ax.bar(x - width/2, initial_values, width, label='Initial', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, final_values, width, label='Final', alpha=0.7, color='orange')
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value (log scale)')
    ax.set_title('Initial vs Final Parameter Values (Log Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yscale('log')  # Use logarithmic scale for y-axis
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_initial_vs_final_log.png'))
    plt.close()
    
    # FIX: Split parameters into groups by magnitude for better visualization
    # Group parameters by orders of magnitude
    small_params = {}
    medium_params = {}
    large_params = {}
    
    for param, details in top_params.items():
        max_val = max(abs(details['initial_value']), abs(details['final_value']))
        if max_val < 10:
            small_params[param] = details
        elif max_val < 100:
            medium_params[param] = details
        else:
            large_params[param] = details
    
    # Create separate charts for each magnitude group
    for group_name, group_params in [
        ('Small (< 10)', small_params),
        ('Medium (10-100)', medium_params),
        ('Large (> 100)', large_params)
    ]:
        if not group_params:
            continue
            
        g_param_names = list(group_params.keys())
        g_initial_values = [group_params[param]['initial_value'] for param in g_param_names]
        g_final_values = [group_params[param]['final_value'] for param in g_param_names]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(g_param_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, g_initial_values, width, label='Initial', alpha=0.7, color='skyblue')
        bars2 = ax.bar(x + width/2, g_final_values, width, label='Final', alpha=0.7, color='orange')
        
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Value')
        ax.set_title(f'Initial vs Final Parameter Values - {group_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(g_param_names, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on top of bars
        for i, bars in enumerate([bars1, bars2]):
            values = g_initial_values if i == 0 else g_final_values
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(g_initial_values + g_final_values),
                      f'{values[j]:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'parameter_initial_vs_final_{group_name.lower().replace(" ", "_").replace("<", "lt").replace("-", "to").replace(">", "gt")}.png'))
        plt.close()

def create_win_rate_visualizations(win_rate_data, output_dir, config):
    """
    Create visualizations for win rate data.
    
    Args:
        win_rate_data: Dictionary with win rate analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'win_rate_history' not in win_rate_data or not win_rate_data['win_rate_history']:
        print("No win rate data available for visualization")
        return
    
    win_rate_history = win_rate_data['win_rate_history']
    generations = [entry['generation'] for entry in win_rate_history]
    
    # Get raw win rates and draw rates
    tiger_win_rates = [entry['tiger_win_rate'] for entry in win_rate_history]
    goat_win_rates = [entry['goat_win_rate'] for entry in win_rate_history]
    draw_rates = [entry['draw_rate'] for entry in win_rate_history]
    
    # Get adjusted win rates
    tiger_adjusted_rates = [entry['tiger_adjusted_win_rate'] for entry in win_rate_history]
    goat_adjusted_rates = [entry['goat_adjusted_win_rate'] for entry in win_rate_history]
    
    # Create raw win rates trend visualization
    figsize = config.get('visualization', {}).get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(generations, tiger_win_rates, 'orange', linewidth=2, marker='o', markersize=4, label='Tiger Win Rate')
    ax.plot(generations, goat_win_rates, 'purple', linewidth=2, marker='s', markersize=4, label='Goat Win Rate')
    ax.plot(generations, draw_rates, 'gray', linewidth=2, marker='^', markersize=4, label='Draw Rate')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Rate')
    ax.set_title('Raw Win Rates Over Generations')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_win_rates_trend.png'))
    plt.close()
    
    # Create adjusted win rates trend visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(generations, tiger_adjusted_rates, 'orange', linewidth=2, marker='o', markersize=4, label='Tiger Adjusted Rate')
    ax.plot(generations, goat_adjusted_rates, 'purple', linewidth=2, marker='s', markersize=4, label='Goat Adjusted Rate')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Rate')
    ax.set_title('Adjusted Win Rates Over Generations (Draws = 0.5)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add a horizontal line at 0.5 to indicate balanced outcomes
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Balance Point')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjusted_win_rates_trend.png'))
    plt.close()
    
    # Create win rate distribution visualization (raw rates)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use average rates for a more accurate representation
    avg_tiger_win_rate = win_rate_data.get('avg_tiger_win_rate', 0)
    avg_goat_win_rate = win_rate_data.get('avg_goat_win_rate', 0)
    avg_draw_rate = win_rate_data.get('avg_draw_rate', 0)
    
    labels = ['Tiger Wins', 'Goat Wins', 'Draws']
    sizes = [avg_tiger_win_rate, avg_goat_win_rate, avg_draw_rate]
    colors = ['orange', 'purple', 'gray']
    explode = (0.1, 0.1, 0) 
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
         shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    ax.set_title('Average Win Rate Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rate_distribution.png'))
    plt.close()
    
    # Create adjusted win rate distribution visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get adjusted win rates
    avg_tiger_adjusted = win_rate_data.get('avg_tiger_adjusted_win_rate', 0)
    avg_goat_adjusted = win_rate_data.get('avg_goat_adjusted_win_rate', 0)
    
    labels = ['Tiger Advantage', 'Goat Advantage']
    sizes = [avg_tiger_adjusted, avg_goat_adjusted]
    colors = ['orange', 'purple']
    explode = (0.1, 0.1)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
         shadow=True, startangle=90)
    ax.axis('equal')
    ax.set_title('Adjusted Win Rate Distribution (Draws Split 0.5/0.5)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adjusted_win_rate_distribution.png'))
    plt.close()

def create_diversity_visualizations(diversity_data, output_dir, config):
    """
    Create visualizations for population diversity data.
    
    Args:
        diversity_data: Dictionary with diversity analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'diversity_history' not in diversity_data or not diversity_data['diversity_history']:
        print("No diversity data available for visualization")
        return
    
    diversity_history = diversity_data['diversity_history']
    generations = [entry['generation'] for entry in diversity_history]
    diversity_values = [entry.get('population_diversity', 0) for entry in diversity_history]
    
    # Create diversity trend visualization
    figsize = config.get('visualization', {}).get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(generations, diversity_values, 'g-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Diversity')
    ax.set_title('Population Diversity Over Generations')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for initial and final diversity
    if len(diversity_values) > 1:
        ax.annotate(f'Initial: {diversity_values[0]:.4f}', 
                  xy=(generations[0], diversity_values[0]),
                  xytext=(generations[0] + 1, diversity_values[0] + 0.02),
                  arrowprops=dict(arrowstyle='->'))
        
        ax.annotate(f'Final: {diversity_values[-1]:.4f}', 
                  xy=(generations[-1], diversity_values[-1]),
                  xytext=(generations[-1] - 3, diversity_values[-1] + 0.02),
                  arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_trend.png'))
    plt.close()

def create_time_efficiency_visualizations(time_data, output_dir, config):
    """
    Create visualizations for time efficiency data.
    
    Args:
        time_data: Dictionary with time efficiency analysis results
        output_dir: Directory to save visualizations
        config: Visualization configuration
    """
    setup_visualization_style(config)
    
    if 'time_history' not in time_data or not time_data['time_history']:
        print("No time efficiency data available for visualization")
        return
    
    time_history = time_data['time_history']
    generations = [entry['generation'] for entry in time_history]
    elapsed_times = [entry['elapsed_time'] for entry in time_history]
    
    # Create time efficiency visualization
    figsize = config.get('visualization', {}).get('figsize', (10, 6))
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(generations, elapsed_times, alpha=0.7, color='blue')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time Per Generation')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add a horizontal line for the average time
    avg_time = time_data['avg_time_per_generation']
    ax.axhline(y=avg_time, color='r', linestyle='--', 
             label=f'Average: {avg_time:.2f}s')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_efficiency.png'))
    plt.close()