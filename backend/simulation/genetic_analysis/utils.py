"""
Utility functions for genetic algorithm analysis.
"""
import os
import json
import csv
import pandas as pd
import glob
from datetime import datetime

def load_config(config_path):
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file (or None to use defaults)
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'analysis': {
            'include_parameter_evolution': True,
            'include_fitness_analysis': True,
            'include_win_rate_analysis': True,
            'include_diversity_analysis': True,
            'include_time_efficiency': True,
            'include_parameter_comparison': True,
            'convergence_threshold': 0.001,
            'stability_count': 3
        },
        'visualization': {
            'dpi': 300,
            'figsize': (10, 6),
            'style': 'seaborn-v0_8-whitegrid',
            'custom_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        },
        'statistics': {
            'significance_threshold': 0.05,
            'confidence_level': 0.95
        }
    }
    
    if config_path is None:
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Merge with default config to ensure all keys exist
        merged_config = default_config.copy()
        for section in loaded_config:
            if section in merged_config:
                merged_config[section].update(loaded_config[section])
            else:
                merged_config[section] = loaded_config[section]
        
        return merged_config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        print(f"Using default configuration")
        return default_config

def load_and_preprocess_data(source_dir):
    """
    Load and preprocess genetic algorithm data.
    
    Args:
        source_dir: Directory containing genetic algorithm data
        
    Returns:
        Dictionary with processed data
    """
    data = {
        'fitness_history': [],
        'parameter_evolution': [],
        'win_rate_history': [],
        'diversity_history': [],
        'time_history': []
    }
    
    # Check if source directory exists
    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        print(f"Source directory {source_dir} not found")
        return data
    
    # Load fitness history
    fitness_history_path = os.path.join(source_dir, "fitness_history.json")
    if os.path.exists(fitness_history_path):
        try:
            with open(fitness_history_path, 'r') as f:
                data['fitness_history'] = json.load(f)
            print(f"Loaded fitness history with {len(data['fitness_history'])} entries")
        except Exception as e:
            print(f"Error loading fitness history: {e}")
    
    # Load GA progress CSV if it exists
    ga_progress_path = os.path.join(source_dir, "ga_progress.csv")
    if os.path.exists(ga_progress_path):
        try:
            df = pd.read_csv(ga_progress_path)
            print(f"Loaded GA progress data with {len(df)} entries")
            
            # Process data into different categories
            
            # Population diversity history
            if 'population_diversity' in df.columns:
                data['diversity_history'] = df[['generation', 'population_diversity']].to_dict('records')
                print(f"Extracted diversity history with {len(data['diversity_history'])} entries")
            
            # Win rate history
            if 'best_tiger_win_rate' in df.columns and 'best_goat_win_rate' in df.columns:
                win_rate_data = []
                for _, row in df.iterrows():
                    win_rate_data.append({
                        'generation': row['generation'],
                        'tiger_win_rate': row['best_tiger_win_rate'],
                        'goat_win_rate': row['best_goat_win_rate']
                    })
                data['win_rate_history'] = win_rate_data
                print(f"Extracted win rate history with {len(win_rate_data)} entries")
            
            # Time history
            if 'elapsed_time' in df.columns and 'total_time' in df.columns:
                time_data = []
                for _, row in df.iterrows():
                    time_data.append({
                        'generation': row['generation'],
                        'elapsed_time': row['elapsed_time'],
                        'total_time': row['total_time']
                    })
                data['time_history'] = time_data
                print(f"Extracted time history with {len(time_data)} entries")
        except Exception as e:
            print(f"Error processing GA progress data: {e}")
    
    # Extract parameter evolution from generation files
    generation_files = glob.glob(os.path.join(source_dir, "generation_*.json"))
    generation_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    parameter_evolution = []
    for gen_file in generation_files:
        try:
            with open(gen_file, 'r') as f:
                gen_data = json.load(f)
                
                # Get the generation number from the filename
                gen_num = int(os.path.basename(gen_file).split('_')[1].split('.')[0])
                
                # Extract best chromosome's parameters
                if 'chromosomes' in gen_data and len(gen_data['chromosomes']) > 0:
                    best_chromosome = gen_data['chromosomes'][0]  # Assuming sorted by fitness
                    
                    if 'genes' in best_chromosome:
                        # Create a parameter entry with generation number
                        param_entry = {'generation': gen_num, 'timestamp': gen_data.get('timestamp', 0)}
                        param_entry.update(best_chromosome['genes'])
                        parameter_evolution.append(param_entry)
        except Exception as e:
            print(f"Error processing generation file {gen_file}: {e}")
    
    if parameter_evolution:
        data['parameter_evolution'] = parameter_evolution
        print(f"Extracted parameter evolution data from {len(parameter_evolution)} generations")
    
    # Load best parameters
    best_params_path = os.path.join(source_dir, "best_params.json")
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
                
                # Extract genes if they exist
                if 'genes' in best_params:
                    data['best_params'] = best_params['genes']
                else:
                    data['best_params'] = best_params
                
                print(f"Loaded best parameters with {len(data['best_params'])} parameters")
        except Exception as e:
            print(f"Error loading best parameters: {e}")
    
    # Get total generations
    data['total_generations'] = 0
    if data['fitness_history']:
        data['total_generations'] = len(data['fitness_history'])
    elif data['parameter_evolution']:
        data['total_generations'] = len(data['parameter_evolution'])
    elif os.path.exists(os.path.join(source_dir, "generation_count.txt")):
        try:
            with open(os.path.join(source_dir, "generation_count.txt"), 'r') as f:
                data['total_generations'] = int(f.read().strip())
                print(f"Loaded generation count: {data['total_generations']}")
        except Exception as e:
            print(f"Error loading generation count: {e}")
    
    return data

def ensure_output_directory(path):
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to the directory
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    os.makedirs(path, exist_ok=True)
    return path

def format_time(seconds):
    """
    Format time in seconds to a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {int(remaining_seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {int(remaining_seconds)}s" 