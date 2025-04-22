"""
Core analysis functions for MCTS vs Minimax competition results.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from statsmodels.stats.multitest import multipletests

def load_config(config_path=None):
    """
    Load analysis configuration or use defaults
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary with configuration parameters
    """
    default_config = {
        "statistical": {
            "significance_threshold": 0.05,
            "confidence_level": 0.95
        },
        "visualization": {
            "color_mcts": "#1f77b4",  # Blue
            "color_minimax": "#ff7f0e",  # Orange
            "color_tiger": "#d62728",  # Red
            "color_goat": "#2ca02c"    # Green
        },
        "top_configs": {
            "count": 3
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults for any missing values
            for section in default_config:
                if section not in config:
                    config[section] = default_config[section]
                else:
                    for key in default_config[section]:
                        if key not in config[section]:
                            config[section][key] = default_config[section][key]
                            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return default_config
    else:
        return default_config

def calculate_confidence_intervals(win_rate, n_games, confidence=0.95):
    """
    Calculate Wilson score interval for win rates
    
    Args:
        win_rate: The win rate or adjusted win rate
        n_games: Number of games played
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple with (lower_bound, upper_bound)
    """
    if n_games == 0:
        return (0, 0)
        
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2/n_games
    center = (win_rate + z**2/(2*n_games)) / denominator
    interval = z * np.sqrt(win_rate * (1 - win_rate) / n_games + z**2/(4*n_games**2)) / denominator
    
    lower_bound = max(0, center - interval)
    upper_bound = min(1, center + interval)
    
    return (lower_bound, upper_bound)

def extract_config_details(config_str):
    """
    Extract configuration details from the JSON string
    
    Args:
        config_str: JSON string with configuration
        
    Returns:
        Dictionary with extracted configuration details
    """
    try:
        config = json.loads(config_str)
        return config
    except Exception as e:
        print(f"Error parsing config JSON: {e}")
        print(f"Problematic JSON: {config_str}")
        return {}

def parse_move_history(move_history):
    """
    Parse the move history string into a structured format
    
    Args:
        move_history: String representation of move history
        
    Returns:
        List of move dictionaries
    """
    if not move_history:
        return []
    
    # Clean up the string - remove quotes, extra spaces, etc.
    move_history = move_history.strip('"\'')
    
    # Split by commas
    moves_str = move_history.split(',')
    
    moves = []
    for move_str in moves_str:
        move_str = move_str.strip()
        if not move_str:
            continue
            
        # Extract move type (p = placement, m = movement)
        if move_str.startswith('p'):
            # Placement move: p<position>
            position = move_str[1:]
            x, y = int(position[0]), int(position[1])
            moves.append({
                'type': 'placement',
                'position': {'x': x, 'y': y}
            })
        elif move_str.startswith('m'):
            # Movement move: m<from><to>[c<capture>]
            # Parse from position (2 digits)
            from_x, from_y = int(move_str[1]), int(move_str[2])
            
            # Parse to position (2 digits)
            to_x, to_y = int(move_str[3]), int(move_str[4])
            
            move = {
                'type': 'movement',
                'from': {'x': from_x, 'y': from_y},
                'to': {'x': to_x, 'y': to_y}
            }
            
            # Check if there's a capture
            if len(move_str) > 5 and 'c' in move_str:
                capture_idx = move_str.find('c')
                if capture_idx > 0 and capture_idx + 2 < len(move_str):
                    capture_x, capture_y = int(move_str[capture_idx+1]), int(move_str[capture_idx+2])
                    move['capture'] = {'x': capture_x, 'y': capture_y}
            
            moves.append(move)
    
    return moves

def load_and_preprocess_data(filepath):
    """
    Load competition results and preprocess for analysis.
    
    Args:
        filepath: Path to the competition results CSV file
        
    Returns:
        DataFrame with preprocessed data
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Extract MCTS and Minimax configuration details
    df['tiger_config_details'] = df['tiger_config'].apply(extract_config_details)
    df['goat_config_details'] = df['goat_config'].apply(extract_config_details)
    
    # Extract algorithm-specific identifiers
    df['tiger_config_id'] = df['tiger_config_details'].apply(lambda x: x.get('config_id', 'unknown'))
    df['goat_config_id'] = df['goat_config_details'].apply(lambda x: x.get('config_id', 'unknown'))
    
    # Extract Minimax depths where applicable
    def extract_depth(config_details):
        if config_details.get('algorithm') == 'minimax':
            return config_details.get('max_depth', 0)
        return None
    
    df['tiger_depth'] = df['tiger_config_details'].apply(extract_depth)
    df['goat_depth'] = df['goat_config_details'].apply(extract_depth)
    
    # Add outcome columns for easier analysis
    df['tiger_won'] = df['winner'] == 'TIGER'
    df['goat_won'] = df['winner'] == 'GOAT'
    df['draw'] = df['winner'] == 'DRAW'
    
    # Parse move histories for detailed analysis
    df['parsed_moves'] = df['move_history'].apply(parse_move_history)
    
    # Calculate derived metrics
    df['adjusted_win_rate_tiger'] = df['tiger_won'].astype(float) + 0.5 * df['draw'].astype(float)
    df['adjusted_win_rate_goat'] = df['goat_won'].astype(float) + 0.5 * df['draw'].astype(float)
    
    # Fill missing values
    df['goats_captured'] = df['goats_captured'].fillna(0).astype(int)
    df['first_capture_move'] = df['first_capture_move'].fillna(-1).astype(int)
    df['phase_transition_move'] = df['phase_transition_move'].fillna(-1).astype(int)
    
    # Add machine-specific normalization factor
    machine_ids = df['machine_id'].unique()
    machine_factors = {}
    
    for machine_id in machine_ids:
        machine_games = df[df['machine_id'] == machine_id]
        # Use average game duration as a proxy for machine speed
        avg_duration = machine_games['game_duration'].mean()
        if avg_duration > 0:
            machine_factors[machine_id] = 1.0  # Base factor, will be adjusted relative to others
    
    # Normalize factors relative to the fastest machine
    if machine_factors:
        min_factor = min(machine_factors.values())
        for machine_id in machine_factors:
            machine_factors[machine_id] = min_factor / machine_factors[machine_id]
    
    # Apply machine factor to the dataframe
    df['machine_factor'] = df['machine_id'].map(machine_factors).fillna(1.0)
    
    return df

def analyze_performance(df, config=None):
    """
    Analyze algorithm performance from the competition data.
    
    Args:
        df: DataFrame with preprocessed competition data
        config: Configuration dictionary with analysis parameters
        
    Returns:
        Dictionary with performance metrics
    """
    # Get how many top configs to include
    top_configs_count = 3  # Default value
    if config and 'top_configs' in config and 'count' in config['top_configs']:
        top_configs_count = config['top_configs']['count']
    
    # Get unique configurations
    mcts_configs = list(set(
        [config for config in df['tiger_config_id'].unique() if 'mcts' in config] + 
        [config for config in df['goat_config_id'].unique() if 'mcts' in config]
    ))
    
    minimax_depths = sorted(list(set(
        [depth for depth in df['tiger_depth'].dropna().unique() if depth > 0] + 
        [depth for depth in df['goat_depth'].dropna().unique() if depth > 0]
    )))
    
    # Calculate overall algorithm performance
    mcts_tiger = df[df['tiger_algorithm'] == 'mcts']
    mcts_goat = df[df['goat_algorithm'] == 'mcts']
    minimax_tiger = df[df['tiger_algorithm'] == 'minimax']
    minimax_goat = df[df['goat_algorithm'] == 'minimax']
    
    # MCTS performance
    mcts_as_tiger_win_rate = mcts_tiger['tiger_won'].mean()
    mcts_as_tiger_draw_rate = mcts_tiger['draw'].mean()
    mcts_as_tiger_adjusted = mcts_as_tiger_win_rate + 0.5 * mcts_as_tiger_draw_rate
    
    mcts_as_goat_win_rate = mcts_goat['goat_won'].mean()
    mcts_as_goat_draw_rate = mcts_goat['draw'].mean()
    mcts_as_goat_adjusted = mcts_as_goat_win_rate + 0.5 * mcts_as_goat_draw_rate
    
    # Minimax performance
    minimax_as_tiger_win_rate = minimax_tiger['tiger_won'].mean()
    minimax_as_tiger_draw_rate = minimax_tiger['draw'].mean()
    minimax_as_tiger_adjusted = minimax_as_tiger_win_rate + 0.5 * minimax_as_tiger_draw_rate
    
    minimax_as_goat_win_rate = minimax_goat['goat_won'].mean()
    minimax_as_goat_draw_rate = minimax_goat['draw'].mean()
    minimax_as_goat_adjusted = minimax_as_goat_win_rate + 0.5 * minimax_as_goat_draw_rate
    
    # Overall metrics
    mcts_games = len(mcts_tiger) + len(mcts_goat)
    mcts_wins = mcts_tiger['tiger_won'].sum() + mcts_goat['goat_won'].sum()
    mcts_draws = mcts_tiger['draw'].sum() + mcts_goat['draw'].sum()
    mcts_overall_win_rate = (mcts_wins + 0.5 * mcts_draws) / mcts_games if mcts_games > 0 else 0
    
    minimax_games = len(minimax_tiger) + len(minimax_goat)
    minimax_wins = minimax_tiger['tiger_won'].sum() + minimax_goat['goat_won'].sum()
    minimax_draws = minimax_tiger['draw'].sum() + minimax_goat['draw'].sum()
    minimax_overall_win_rate = (minimax_wins + 0.5 * minimax_draws) / minimax_games if minimax_games > 0 else 0
    
    draw_rate = (mcts_tiger['draw'].sum() + minimax_tiger['draw'].sum()) / len(df)
    
    # Calculate confidence intervals
    mcts_as_tiger_ci = calculate_confidence_intervals(mcts_as_tiger_adjusted, len(mcts_tiger))
    mcts_as_goat_ci = calculate_confidence_intervals(mcts_as_goat_adjusted, len(mcts_goat))
    minimax_as_tiger_ci = calculate_confidence_intervals(minimax_as_tiger_adjusted, len(minimax_tiger))
    minimax_as_goat_ci = calculate_confidence_intervals(minimax_as_goat_adjusted, len(minimax_goat))
    mcts_overall_ci = calculate_confidence_intervals(mcts_overall_win_rate, mcts_games)
    minimax_overall_ci = calculate_confidence_intervals(minimax_overall_win_rate, minimax_games)
    
    # Create algorithm comparison dataframe
    algorithm_comparison = pd.DataFrame({
        'Algorithm': ['MCTS', 'Minimax'],
        'Games': [mcts_games, minimax_games],
        'Win Rate': [mcts_overall_win_rate, minimax_overall_win_rate],
        'CI Lower': [mcts_overall_ci[0], minimax_overall_ci[0]],
        'CI Upper': [mcts_overall_ci[1], minimax_overall_ci[1]],
        'As Tiger Win Rate': [mcts_as_tiger_adjusted, minimax_as_tiger_adjusted],
        'As Tiger CI Lower': [mcts_as_tiger_ci[0], minimax_as_tiger_ci[0]],
        'As Tiger CI Upper': [mcts_as_tiger_ci[1], minimax_as_tiger_ci[1]],
        'As Goat Win Rate': [mcts_as_goat_adjusted, minimax_as_goat_adjusted],
        'As Goat CI Lower': [mcts_as_goat_ci[0], minimax_as_goat_ci[0]],
        'As Goat CI Upper': [mcts_as_goat_ci[1], minimax_as_goat_ci[1]]
    })
    
    # Calculate wins and draws for each algorithm and role combination
    mcts_tiger_wins = mcts_tiger['tiger_won'].sum()
    mcts_tiger_draws = mcts_tiger['draw'].sum()
    mcts_goat_wins = mcts_goat['goat_won'].sum()
    mcts_goat_draws = mcts_goat['draw'].sum()
    minimax_tiger_wins = minimax_tiger['tiger_won'].sum()
    minimax_tiger_draws = minimax_tiger['draw'].sum()
    minimax_goat_wins = minimax_goat['goat_won'].sum()
    minimax_goat_draws = minimax_goat['draw'].sum()

    # Analyze algorithm and role performance
    roles = ['MCTS as Tiger', 'MCTS as Goat', 'Minimax as Tiger', 'Minimax as Goat']
    
    role_data = {
        'Role': roles,
        'Games': [len(mcts_tiger), len(mcts_goat), len(minimax_tiger), len(minimax_goat)],
        'Wins': [mcts_tiger_wins, mcts_goat_wins, minimax_tiger_wins, minimax_goat_wins],
        'Draws': [mcts_tiger_draws, mcts_goat_draws, minimax_tiger_draws, minimax_goat_draws],
        'Losses': [len(mcts_tiger) - mcts_tiger_wins - mcts_tiger_draws,
                  len(mcts_goat) - mcts_goat_wins - mcts_goat_draws,
                  len(minimax_tiger) - minimax_tiger_wins - minimax_tiger_draws,
                  len(minimax_goat) - minimax_goat_wins - minimax_goat_draws]
    }
    
    # Calculate percentages
    role_data['Win %'] = [wins / games if games > 0 else 0 for wins, games in zip(role_data['Wins'], role_data['Games'])]
    role_data['Draw %'] = [draws / games if games > 0 else 0 for draws, games in zip(role_data['Draws'], role_data['Games'])]
    role_data['Loss %'] = [losses / games if games > 0 else 0 for losses, games in zip(role_data['Losses'], role_data['Games'])]
    
    # Calculate adjusted win rate (counting draws as 0.5 wins)
    role_data['Win Rate'] = [(wins + 0.5 * draws) / games if games > 0 else 0 
                            for wins, draws, games in zip(role_data['Wins'], role_data['Draws'], role_data['Games'])]
    
    # Calculate confidence intervals
    role_data['CI Lower'] = [calculate_confidence_intervals(wr, g)[0] if g > 0 else 0 
                            for wr, g in zip(role_data['Win Rate'], role_data['Games'])]
    role_data['CI Upper'] = [calculate_confidence_intervals(wr, g)[1] if g > 0 else 0 
                            for wr, g in zip(role_data['Win Rate'], role_data['Games'])]
    
    role_performance = pd.DataFrame(role_data)
    
    # Analyze Minimax depth performance
    depth_performance_list = []
    
    for depth in minimax_depths:
        # Minimax as Tiger at this depth
        tiger_games = df[(df['tiger_algorithm'] == 'minimax') & (df['tiger_depth'] == depth)]
        tiger_win_rate = tiger_games['tiger_won'].mean() if len(tiger_games) > 0 else 0
        tiger_draw_rate = tiger_games['draw'].mean() if len(tiger_games) > 0 else 0
        tiger_adjusted = tiger_win_rate + 0.5 * tiger_draw_rate
        
        # Minimax as Goat at this depth
        goat_games = df[(df['goat_algorithm'] == 'minimax') & (df['goat_depth'] == depth)]
        goat_win_rate = goat_games['goat_won'].mean() if len(goat_games) > 0 else 0
        goat_draw_rate = goat_games['draw'].mean() if len(goat_games) > 0 else 0
        goat_adjusted = goat_win_rate + 0.5 * goat_draw_rate
        
        # Overall at this depth
        total_games = len(tiger_games) + len(goat_games)
        total_wins = tiger_games['tiger_won'].sum() + goat_games['goat_won'].sum()
        total_draws = tiger_games['draw'].sum() + goat_games['draw'].sum()
        overall_adjusted = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0
        
        # Average move time for this depth
        avg_move_time_tiger = tiger_games['avg_tiger_move_time'].mean() if len(tiger_games) > 0 else 0
        avg_move_time_goat = goat_games['avg_goat_move_time'].mean() if len(goat_games) > 0 else 0
        avg_move_time = (avg_move_time_tiger * len(tiger_games) + 
                         avg_move_time_goat * len(goat_games)) / total_games if total_games > 0 else 0
        
        # Confidence intervals
        tiger_ci = calculate_confidence_intervals(tiger_adjusted, len(tiger_games))
        goat_ci = calculate_confidence_intervals(goat_adjusted, len(goat_games))
        overall_ci = calculate_confidence_intervals(overall_adjusted, total_games)
        
        depth_performance_list.append({
            'Depth': depth,
            'Games': total_games,
            'Win Rate': overall_adjusted,
            'CI Lower': overall_ci[0],
            'CI Upper': overall_ci[1],
            'As Tiger Win Rate': tiger_adjusted,
            'As Tiger CI Lower': tiger_ci[0],
            'As Tiger CI Upper': tiger_ci[1],
            'As Goat Win Rate': goat_adjusted,
            'As Goat CI Lower': goat_ci[0],
            'As Goat CI Upper': goat_ci[1],
            'Avg Move Time (s)': avg_move_time
        })
    
    depth_performance = pd.DataFrame(depth_performance_list)
    
    # Analyze configuration matchups
    config_matchups_list = []
    
    for mcts_config in mcts_configs:
        for depth in minimax_depths:
            # MCTS as Tiger vs Minimax as Goat
            tiger_games = df[
                (df['tiger_algorithm'] == 'mcts') & 
                (df['tiger_config_id'] == mcts_config) & 
                (df['goat_algorithm'] == 'minimax') & 
                (df['goat_depth'] == depth)
            ]
            
            # MCTS as Goat vs Minimax as Tiger
            goat_games = df[
                (df['goat_algorithm'] == 'mcts') & 
                (df['goat_config_id'] == mcts_config) & 
                (df['tiger_algorithm'] == 'minimax') & 
                (df['tiger_depth'] == depth)
            ]
            
            if len(tiger_games) > 0 or len(goat_games) > 0:
                # MCTS as Tiger performance
                tiger_win_rate = tiger_games['tiger_won'].mean() if len(tiger_games) > 0 else 0
                tiger_draw_rate = tiger_games['draw'].mean() if len(tiger_games) > 0 else 0
                tiger_adjusted = tiger_win_rate + 0.5 * tiger_draw_rate
                
                # MCTS as Goat performance
                goat_win_rate = goat_games['goat_won'].mean() if len(goat_games) > 0 else 0
                goat_draw_rate = goat_games['draw'].mean() if len(goat_games) > 0 else 0
                goat_adjusted = goat_win_rate + 0.5 * goat_draw_rate
                
                # Overall MCTS performance in this matchup
                total_games = len(tiger_games) + len(goat_games)
                mcts_wins = tiger_games['tiger_won'].sum() + goat_games['goat_won'].sum()
                mcts_draws = tiger_games['draw'].sum() + goat_games['draw'].sum()
                overall_adjusted = (mcts_wins + 0.5 * mcts_draws) / total_games
                
                # Confidence intervals
                tiger_ci = calculate_confidence_intervals(tiger_adjusted, len(tiger_games))
                goat_ci = calculate_confidence_intervals(goat_adjusted, len(goat_games))
                overall_ci = calculate_confidence_intervals(overall_adjusted, total_games)
                
                config_matchups_list.append({
                    'MCTS Config': mcts_config,
                    'Minimax Depth': depth,
                    'Games': total_games,
                    'Win Rate': overall_adjusted,
                    'CI Lower': overall_ci[0],
                    'CI Upper': overall_ci[1],
                    'As Tiger Win Rate': tiger_adjusted,
                    'As Tiger Games': len(tiger_games),
                    'As Tiger CI Lower': tiger_ci[0],
                    'As Tiger CI Upper': tiger_ci[1],
                    'As Goat Win Rate': goat_adjusted,
                    'As Goat Games': len(goat_games),
                    'As Goat CI Lower': goat_ci[0],
                    'As Goat CI Upper': goat_ci[1]
                })
    
    config_matchups = pd.DataFrame(config_matchups_list)
    
    # Find top performing configurations
    mcts_performance = []
    for config in mcts_configs:
        config_games = df[(df['tiger_algorithm'] == 'mcts') & (df['tiger_config_id'] == config) | 
                          (df['goat_algorithm'] == 'mcts') & (df['goat_config_id'] == config)]
        
        if len(config_games) > 0:
            # Games as Tiger
            tiger_games = config_games[config_games['tiger_config_id'] == config]
            tiger_wins = tiger_games['tiger_won'].sum()
            tiger_draws = tiger_games['draw'].sum()
            
            # Games as Goat
            goat_games = config_games[config_games['goat_config_id'] == config]
            goat_wins = goat_games['goat_won'].sum()
            goat_draws = goat_games['draw'].sum()
            
            # Overall performance
            total_games = len(tiger_games) + len(goat_games)
            total_wins = tiger_wins + goat_wins
            total_draws = tiger_draws + goat_draws
            win_rate = (total_wins + 0.5 * total_draws) / total_games
            
            mcts_performance.append({
                'config_id': config,
                'games': total_games,
                'win_rate': win_rate,
                'tiger_win_rate': tiger_wins / len(tiger_games) if len(tiger_games) > 0 else 0,
                'goat_win_rate': goat_wins / len(goat_games) if len(goat_games) > 0 else 0
            })
    
    # Sort by win rate
    if top_configs_count <= 0:
        # Include all configs when top_configs_count is 0 or negative
        top_mcts_configs = sorted(mcts_performance, key=lambda x: x['win_rate'], reverse=True)
    else:
        top_mcts_configs = sorted(mcts_performance, key=lambda x: x['win_rate'], reverse=True)[:top_configs_count]
    
    # Top minimax depths
    minimax_performance = []
    for depth in minimax_depths:
        depth_games = df[(df['tiger_algorithm'] == 'minimax') & (df['tiger_depth'] == depth) | 
                         (df['goat_algorithm'] == 'minimax') & (df['goat_depth'] == depth)]
        
        if len(depth_games) > 0:
            # Games as Tiger
            tiger_games = depth_games[depth_games['tiger_depth'] == depth]
            tiger_wins = tiger_games['tiger_won'].sum()
            tiger_draws = tiger_games['draw'].sum()
            
            # Games as Goat
            goat_games = depth_games[depth_games['goat_depth'] == depth]
            goat_wins = goat_games['goat_won'].sum()
            goat_draws = goat_games['draw'].sum()
            
            # Overall performance
            total_games = len(tiger_games) + len(goat_games)
            total_wins = tiger_wins + goat_wins
            total_draws = tiger_draws + goat_draws
            win_rate = (total_wins + 0.5 * total_draws) / total_games
            
            # Average move time
            avg_move_time_tiger = tiger_games['avg_tiger_move_time'].mean() if len(tiger_games) > 0 else 0
            avg_move_time_goat = goat_games['avg_goat_move_time'].mean() if len(goat_games) > 0 else 0
            avg_move_time = (avg_move_time_tiger * len(tiger_games) + 
                             avg_move_time_goat * len(goat_games)) / total_games
            
            minimax_performance.append({
                'depth': depth,
                'games': total_games,
                'win_rate': win_rate,
                'tiger_win_rate': tiger_wins / len(tiger_games) if len(tiger_games) > 0 else 0,
                'goat_win_rate': goat_wins / len(goat_games) if len(goat_games) > 0 else 0,
                'avg_move_time': avg_move_time
            })
    
    # Sort by win rate
    if top_configs_count <= 0:
        # Include all configs when top_configs_count is 0 or negative
        top_minimax_configs = sorted(minimax_performance, key=lambda x: x['win_rate'], reverse=True)
    else:
        top_minimax_configs = sorted(minimax_performance, key=lambda x: x['win_rate'], reverse=True)[:top_configs_count]
    
    return {
        'algorithm_comparison': algorithm_comparison,
        'role_performance': role_performance,
        'depth_performance': depth_performance,
        'config_matchups': config_matchups,
        'mcts_configs': mcts_configs,
        'minimax_depths': minimax_depths,
        'mcts_overall_win_rate': mcts_overall_win_rate,
        'minimax_overall_win_rate': minimax_overall_win_rate,
        'draw_rate': draw_rate,
        'mcts_as_tiger_win_rate': mcts_as_tiger_adjusted,
        'mcts_as_goat_win_rate': mcts_as_goat_adjusted,
        'minimax_as_tiger_win_rate': minimax_as_tiger_adjusted,
        'minimax_as_goat_win_rate': minimax_as_goat_adjusted,
        'top_mcts_configs': top_mcts_configs,
        'top_minimax_configs': top_minimax_configs
    }

def analyze_game_dynamics(df):
    """
    Analyze game dynamics from the competition data.
    
    Args:
        df: DataFrame with preprocessed competition data
        
    Returns:
        Dictionary with game dynamics metrics
    """
    # Calculate basic game length statistics
    avg_game_length = df['moves'].mean()
    avg_length_tiger_win = df[df['tiger_won']]['moves'].mean()
    avg_length_goat_win = df[df['goat_won']]['moves'].mean()
    avg_length_draw = df[df['draw']]['moves'].mean()
    
    # Calculate capture statistics
    capture_counts = df['goats_captured'].value_counts().sort_index()
    avg_captures = df['goats_captured'].mean()
    
    # Goat comeback wins (wins despite captures)
    goat_wins_with_captures = df[(df['goat_won']) & (df['goats_captured'] > 0)]
    goat_wins_by_captures = goat_wins_with_captures.groupby('goats_captured').size()
    
    # Ensure all capture counts from 1 to 4 are represented
    for i in range(1, 5):
        if i not in goat_wins_by_captures.index:
            goat_wins_by_captures[i] = 0
    
    goat_wins_by_captures = goat_wins_by_captures.sort_index()
    
    # First capture timing
    games_with_captures = df[df['first_capture_move'] > 0]
    avg_first_capture = games_with_captures['first_capture_move'].mean()
    
    # Compile capture analysis dataframe
    capture_analysis_data = []
    
    for captures in range(0, 6):  # 0 to 5 captures
        games = df[df['goats_captured'] == captures]
        
        if len(games) > 0:
            tiger_wins = games['tiger_won'].sum()
            goat_wins = games['goat_won'].sum()
            draws = games['draw'].sum()
            
            # Win rates
            tiger_win_rate = tiger_wins / len(games)
            goat_win_rate = goat_wins / len(games)
            draw_rate = draws / len(games)
            
            # Game length
            avg_length = games['moves'].mean()
            
            # First capture timing (when captures > 0)
            avg_first_cap = games[games['first_capture_move'] > 0]['first_capture_move'].mean()
            
            capture_analysis_data.append({
                'Captures': captures,
                'Games': len(games),
                'Tiger Win %': tiger_win_rate * 100,
                'Goat Win %': goat_win_rate * 100,
                'Draw %': draw_rate * 100,
                'Avg Game Length': avg_length,
                'Avg First Capture Move': avg_first_cap if not pd.isna(avg_first_cap) else None
            })
    
    capture_analysis = pd.DataFrame(capture_analysis_data)
    
    # Analyze comeback victories in more detail
    comeback_analysis_data = []
    
    for captures in range(1, 5):  # 1 to 4 captures (goat wins with captures)
        comeback_games = df[(df['goat_won']) & (df['goats_captured'] == captures)]
        
        if len(comeback_games) > 0:
            # By algorithm
            mcts_comebacks = comeback_games[comeback_games['goat_algorithm'] == 'mcts']
            minimax_comebacks = comeback_games[comeback_games['goat_algorithm'] == 'minimax']
            
            # First capture timing
            avg_first_capture = comeback_games['first_capture_move'].mean()
            
            # Game length
            avg_length = comeback_games['moves'].mean()
            
            comeback_analysis_data.append({
                'Captures': captures,
                'Games': len(comeback_games),
                'MCTS Games': len(mcts_comebacks),
                'Minimax Games': len(minimax_comebacks),
                'MCTS %': len(mcts_comebacks) / len(comeback_games) * 100 if len(comeback_games) > 0 else 0,
                'Minimax %': len(minimax_comebacks) / len(comeback_games) * 100 if len(comeback_games) > 0 else 0,
                'Avg First Capture': avg_first_capture,
                'Avg Game Length': avg_length
            })
    
    comeback_analysis = pd.DataFrame(comeback_analysis_data)
    
    # Game length analysis
    length_bins = list(range(0, 101, 10))  # 0-10, 10-20, ..., 90-100
    length_analysis_data = []
    
    for i in range(len(length_bins) - 1):
        min_length = length_bins[i]
        max_length = length_bins[i+1]
        
        bin_games = df[(df['moves'] >= min_length) & (df['moves'] < max_length)]
        
        if len(bin_games) > 0:
            tiger_wins = bin_games['tiger_won'].sum()
            goat_wins = bin_games['goat_won'].sum()
            draws = bin_games['draw'].sum()
            
            # Win rates
            tiger_win_rate = tiger_wins / len(bin_games)
            goat_win_rate = goat_wins / len(bin_games)
            draw_rate = draws / len(bin_games)
            
            # Captures
            avg_captures = bin_games['goats_captured'].mean()
            
            length_analysis_data.append({
                'Length Range': f"{min_length}-{max_length}",
                'Games': len(bin_games),
                'Tiger Win %': tiger_win_rate * 100,
                'Goat Win %': goat_win_rate * 100,
                'Draw %': draw_rate * 100,
                'Avg Captures': avg_captures
            })
    
    length_analysis = pd.DataFrame(length_analysis_data)
    
    # Analyze threefold repetition draws
    repetition_draws = df[(df['draw']) & (df['reason'] == 'THREEFOLD_REPETITION')]
    
    repetition_by_algorithm = {
        'MCTS as Tiger': len(repetition_draws[repetition_draws['tiger_algorithm'] == 'mcts']),
        'MCTS as Goat': len(repetition_draws[repetition_draws['goat_algorithm'] == 'mcts']),
        'Minimax as Tiger': len(repetition_draws[repetition_draws['tiger_algorithm'] == 'minimax']),
        'Minimax as Goat': len(repetition_draws[repetition_draws['goat_algorithm'] == 'minimax'])
    }
    
    repetition_analysis_data = [{
        'Draw Type': 'Threefold Repetition',
        'Games': len(repetition_draws),
        'Avg Length': repetition_draws['moves'].mean() if len(repetition_draws) > 0 else 0,
        'Avg Captures': repetition_draws['goats_captured'].mean() if len(repetition_draws) > 0 else 0,
        'MCTS as Tiger': repetition_by_algorithm['MCTS as Tiger'],
        'MCTS as Goat': repetition_by_algorithm['MCTS as Goat'],
        'Minimax as Tiger': repetition_by_algorithm['Minimax as Tiger'],
        'Minimax as Goat': repetition_by_algorithm['Minimax as Goat']
    }]
    
    repetition_analysis = pd.DataFrame(repetition_analysis_data)
    
    # Analyze decision time (focusing on Minimax)
    time_analysis_data = []
    
    for depth in sorted(df['tiger_depth'].dropna().unique()):
        if depth > 0:  # Valid Minimax depth
            # Minimax as Tiger
            tiger_games = df[(df['tiger_algorithm'] == 'minimax') & (df['tiger_depth'] == depth)]
            if len(tiger_games) > 0:
                time_analysis_data.append({
                    'Role': 'Tiger',
                    'Depth': depth,
                    'Avg Move Time (s)': tiger_games['avg_tiger_move_time'].mean(),
                    'Games': len(tiger_games)
                })
            
            # Minimax as Goat
            goat_games = df[(df['goat_algorithm'] == 'minimax') & (df['goat_depth'] == depth)]
            if len(goat_games) > 0:
                time_analysis_data.append({
                    'Role': 'Goat',
                    'Depth': depth,
                    'Avg Move Time (s)': goat_games['avg_goat_move_time'].mean(),
                    'Games': len(goat_games)
                })
    
    time_analysis = pd.DataFrame(time_analysis_data)
    
    return {
        'capture_analysis': capture_analysis,
        'comeback_analysis': comeback_analysis,
        'length_analysis': length_analysis,
        'repetition_analysis': repetition_analysis,
        'time_analysis': time_analysis,
        'avg_game_length': avg_game_length,
        'avg_length_tiger_win': avg_length_tiger_win,
        'avg_length_goat_win': avg_length_goat_win,
        'avg_length_draw': avg_length_draw,
        'avg_captures': avg_captures,
        'avg_first_capture': avg_first_capture,
        'goat_wins_1_capture': goat_wins_by_captures.get(1, 0),
        'goat_wins_2_capture': goat_wins_by_captures.get(2, 0),
        'goat_wins_3_capture': goat_wins_by_captures.get(3, 0),
        'goat_wins_4_capture': goat_wins_by_captures.get(4, 0)
    }

def analyze_movement_patterns(df):
    """
    Analyze specific movement patterns from the competition data.
    
    Args:
        df: DataFrame with preprocessed competition data
        
    Returns:
        Dictionary with movement pattern metrics
    """
    # Analyze tiger's 2nd-ply response patterns
    opening_responses = {}
    
    # Board positions for easy reference
    board_positions = {}
    for x in range(5):
        for y in range(5):
            board_positions[(x, y)] = f"({x},{y})"
    
    # Extract games with at least 2 moves
    games_with_opening = df[df['moves'] >= 2]
    
    for _, game in games_with_opening.iterrows():
        parsed_moves = game['parsed_moves']
        
        if len(parsed_moves) >= 2:
            # First move is goat's placement
            goat_opening = None
            if parsed_moves[0]['type'] == 'placement':
                pos = parsed_moves[0]['position']
                goat_opening = (pos['x'], pos['y'])
            
            # Second move is tiger's response
            tiger_response = None
            if len(parsed_moves) > 1 and parsed_moves[1]['type'] == 'movement':
                move = parsed_moves[1]
                from_pos = (move['from']['x'], move['from']['y'])
                to_pos = (move['to']['x'], move['to']['y'])
                tiger_response = (from_pos, to_pos)
            
            if goat_opening and tiger_response:
                # Create key for this opening pattern
                opening_key = f"Goat @ {board_positions[goat_opening]}"
                response_key = f"{board_positions[tiger_response[0]]} → {board_positions[tiger_response[1]]}"
                
                # Initialize data structures if needed
                if opening_key not in opening_responses:
                    opening_responses[opening_key] = {
                        'total': 0,
                        'responses': {},
                        'tiger_wins': 0,
                        'goat_wins': 0,
                        'draws': 0
                    }
                
                if response_key not in opening_responses[opening_key]['responses']:
                    opening_responses[opening_key]['responses'][response_key] = {
                        'total': 0,
                        'tiger_wins': 0,
                        'goat_wins': 0,
                        'draws': 0
                    }
                
                # Update counts
                opening_responses[opening_key]['total'] += 1
                opening_responses[opening_key]['responses'][response_key]['total'] += 1
                
                # Update outcomes
                if game['tiger_won']:
                    opening_responses[opening_key]['tiger_wins'] += 1
                    opening_responses[opening_key]['responses'][response_key]['tiger_wins'] += 1
                elif game['goat_won']:
                    opening_responses[opening_key]['goat_wins'] += 1
                    opening_responses[opening_key]['responses'][response_key]['goat_wins'] += 1
                else:  # Draw
                    opening_responses[opening_key]['draws'] += 1
                    opening_responses[opening_key]['responses'][response_key]['draws'] += 1
    
    # Create a structured dataframe for opening analysis
    opening_analysis_data = []
    
    for opening, opening_data in opening_responses.items():
        # Calculate win rates for this opening
        tiger_win_rate = opening_data['tiger_wins'] / opening_data['total'] if opening_data['total'] > 0 else 0
        goat_win_rate = opening_data['goat_wins'] / opening_data['total'] if opening_data['total'] > 0 else 0
        draw_rate = opening_data['draws'] / opening_data['total'] if opening_data['total'] > 0 else 0
        
        # Get top responses by frequency
        responses_sorted = sorted(
            opening_data['responses'].items(), 
            key=lambda x: x[1]['total'], 
            reverse=True
        )
        
        for response_key, response_data in responses_sorted[:3]:  # Top 3 responses
            response_tiger_win_rate = response_data['tiger_wins'] / response_data['total'] if response_data['total'] > 0 else 0
            response_goat_win_rate = response_data['goat_wins'] / response_data['total'] if response_data['total'] > 0 else 0
            response_draw_rate = response_data['draws'] / response_data['total'] if response_data['total'] > 0 else 0
            
            opening_analysis_data.append({
                'Opening': opening,
                'Response': response_key,
                'Games': response_data['total'],
                'Tiger Win %': response_tiger_win_rate * 100,
                'Goat Win %': response_goat_win_rate * 100,
                'Draw %': response_draw_rate * 100,
                'Opening Total Games': opening_data['total'],
                'Opening Tiger Win %': tiger_win_rate * 100,
                'Opening Goat Win %': goat_win_rate * 100,
                'Opening Draw %': draw_rate * 100
            })
    
    opening_analysis = pd.DataFrame(opening_analysis_data)
    
    # Analyze capture patterns - where captures most commonly occur
    capture_positions = {}
    
    for _, game in df.iterrows():
        parsed_moves = game['parsed_moves']
        
        for move in parsed_moves:
            if move['type'] == 'movement' and 'capture' in move:
                capture_pos = (move['capture']['x'], move['capture']['y'])
                
                if capture_pos not in capture_positions:
                    capture_positions[capture_pos] = 0
                
                capture_positions[capture_pos] += 1
    
    # Create a structured dataframe for capture pattern analysis
    capture_pattern_data = []
    
    for pos, count in sorted(capture_positions.items(), key=lambda x: x[1], reverse=True):
        capture_pattern_data.append({
            'Position': board_positions[pos],
            'X': pos[0],
            'Y': pos[1],
            'Count': count,
            'Percentage': count / sum(capture_positions.values()) * 100
        })
    
    capture_pattern_analysis = pd.DataFrame(capture_pattern_data)
    
    return {
        'opening_analysis': opening_analysis,
        'capture_pattern_analysis': capture_pattern_analysis
    }

def check_test_assumptions(data_groups):
    """
    Perform basic checks on statistical test assumptions.
    
    Args:
        data_groups: List of data arrays to check
        
    Returns:
        Dictionary with assumption check results
    """
    results = {}
    
    # Check sample sizes
    sample_sizes = [len(group) for group in data_groups if hasattr(group, '__len__')]
    results['sample_sizes'] = sample_sizes
    
    # Check if samples are large enough for CLT to apply
    results['adequate_sample_size'] = all(size >= 30 for size in sample_sizes)
    
    # Basic normality check (simplified)
    try:
        normality_results = []
        for i, group in enumerate(data_groups):
            if hasattr(group, '__len__') and len(group) >= 8:  # Minimum for Shapiro-Wilk
                stat, p = stats.shapiro(group)
                normality_results.append({
                    'group': i,
                    'p_value': p,
                    'normal': p > 0.05
                })
        results['normality'] = normality_results
    except Exception as e:
        results['normality'] = f"Error checking normality: {str(e)}"
    
    # Check homogeneity of variance
    try:
        if len(data_groups) == 2 and all(hasattr(group, '__len__') for group in data_groups):
            # Use Levene's test for equal variances
            stat, p = stats.levene(data_groups[0], data_groups[1])
            results['homogeneity_of_variance'] = {
                'test': 'Levene',
                'statistic': stat,
                'p_value': p,
                'equal_variance': p > 0.05
            }
    except Exception as e:
        results['homogeneity_of_variance'] = f"Error checking homogeneity of variance: {str(e)}"
    
    return results

def perform_statistical_tests(df):
    """
    Perform statistical tests on the competition data.
    
    Args:
        df: DataFrame with preprocessed competition data
        
    Returns:
        Dictionary with statistical test results
    """
    results = {}
    
    # Compare MCTS vs Minimax overall performance
    mcts_tiger = df[df['tiger_algorithm'] == 'mcts']
    mcts_goat = df[df['goat_algorithm'] == 'mcts']
    minimax_tiger = df[df['tiger_algorithm'] == 'minimax']
    minimax_goat = df[df['goat_algorithm'] == 'minimax']
    
    # Calculate overall adjusted win rates (counting draws as 0.5)
    mcts_results = np.concatenate([
        mcts_tiger['tiger_won'].astype(float) + 0.5 * mcts_tiger['draw'].astype(float),
        mcts_goat['goat_won'].astype(float) + 0.5 * mcts_goat['draw'].astype(float)
    ])
    
    minimax_results = np.concatenate([
        minimax_tiger['tiger_won'].astype(float) + 0.5 * minimax_tiger['draw'].astype(float),
        minimax_goat['goat_won'].astype(float) + 0.5 * minimax_goat['draw'].astype(float)
    ])
    
    # Check assumptions for algorithm comparison
    assumption_checks = check_test_assumptions([mcts_results, minimax_results])
    results['assumption_checks'] = assumption_checks
    
    # Perform t-test comparing MCTS and Minimax performance
    t_stat, p_value = stats.ttest_ind(mcts_results, minimax_results, equal_var=False)
    
    results['algorithm_comparison_test'] = {
        'statistic': t_stat,
        'p_value': p_value,
        'mcts_mean': np.mean(mcts_results),
        'minimax_mean': np.mean(minimax_results),
        'mcts_std': np.std(mcts_results, ddof=1),
        'minimax_std': np.std(minimax_results, ddof=1),
        'mcts_n': len(mcts_results),
        'minimax_n': len(minimax_results)
    }
    
    # Calculate effect size
    effect_size = abs(np.mean(mcts_results) - np.mean(minimax_results))
    results['algorithm_comparison_test']['effect_size'] = effect_size
    
    # Compare Minimax depths pairwise
    depth_comparison_tests = {}
    minimax_depths = sorted(df['tiger_depth'].dropna().unique())
    
    for i, depth1 in enumerate(minimax_depths):
        for depth2 in minimax_depths[i+1:]:
            # Skip invalid depths
            if depth1 <= 0 or depth2 <= 0:
                continue
                
            # Get games with the specified depths
            depth1_tiger = df[(df['tiger_algorithm'] == 'minimax') & (df['tiger_depth'] == depth1)]
            depth1_goat = df[(df['goat_algorithm'] == 'minimax') & (df['goat_depth'] == depth1)]
            depth2_tiger = df[(df['tiger_algorithm'] == 'minimax') & (df['tiger_depth'] == depth2)]
            depth2_goat = df[(df['goat_algorithm'] == 'minimax') & (df['goat_depth'] == depth2)]
            
            # Calculate adjusted win rates
            depth1_results = np.concatenate([
                depth1_tiger['tiger_won'].astype(float) + 0.5 * depth1_tiger['draw'].astype(float),
                depth1_goat['goat_won'].astype(float) + 0.5 * depth1_goat['draw'].astype(float)
            ])
            
            depth2_results = np.concatenate([
                depth2_tiger['tiger_won'].astype(float) + 0.5 * depth2_tiger['draw'].astype(float),
                depth2_goat['goat_won'].astype(float) + 0.5 * depth2_goat['draw'].astype(float)
            ])
            
            # Check assumptions
            depth_assumptions = check_test_assumptions([depth1_results, depth2_results])
            
            # Perform t-test
            if len(depth1_results) > 0 and len(depth2_results) > 0:
                t_stat, p_value = stats.ttest_ind(depth1_results, depth2_results, equal_var=False)
                
                # Calculate effect size
                effect_size = abs(np.mean(depth1_results) - np.mean(depth2_results))
                
                depth_comparison_tests[f"{int(depth1)}_vs_{int(depth2)}"] = {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'depth1_mean': np.mean(depth1_results),
                    'depth2_mean': np.mean(depth2_results),
                    'depth1_std': np.std(depth1_results, ddof=1) if len(depth1_results) > 1 else 0,
                    'depth2_std': np.std(depth2_results, ddof=1) if len(depth2_results) > 1 else 0,
                    'depth1_n': len(depth1_results),
                    'depth2_n': len(depth2_results),
                    'effect_size': effect_size,
                    'assumptions': depth_assumptions
                }
    
    results['depth_comparison_tests'] = depth_comparison_tests
    
    # Compare MCTS configurations
    mcts_configs = list(set(
        [config for config in df['tiger_config_id'].unique() if isinstance(config, str) and 'mcts' in config] + 
        [config for config in df['goat_config_id'].unique() if isinstance(config, str) and 'mcts' in config]
    ))
    
    config_comparison_tests = {}
    
    for i, config1 in enumerate(mcts_configs):
        for config2 in mcts_configs[i+1:]:
            # Get games with the specified configurations
            config1_tiger = df[(df['tiger_algorithm'] == 'mcts') & (df['tiger_config_id'] == config1)]
            config1_goat = df[(df['goat_algorithm'] == 'mcts') & (df['goat_config_id'] == config1)]
            config2_tiger = df[(df['tiger_algorithm'] == 'mcts') & (df['tiger_config_id'] == config2)]
            config2_goat = df[(df['goat_algorithm'] == 'mcts') & (df['goat_config_id'] == config2)]
            
            # Calculate adjusted win rates
            config1_results = np.concatenate([
                config1_tiger['tiger_won'].astype(float) + 0.5 * config1_tiger['draw'].astype(float),
                config1_goat['goat_won'].astype(float) + 0.5 * config1_goat['draw'].astype(float)
            ])
            
            config2_results = np.concatenate([
                config2_tiger['tiger_won'].astype(float) + 0.5 * config2_tiger['draw'].astype(float),
                config2_goat['goat_won'].astype(float) + 0.5 * config2_goat['draw'].astype(float)
            ])
            
            # Perform t-test
            if len(config1_results) > 0 and len(config2_results) > 0:
                t_stat, p_value = stats.ttest_ind(config1_results, config2_results, equal_var=False)
                
                config_comparison_tests[f"{config1}_vs_{config2}"] = {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'config1_mean': np.mean(config1_results),
                    'config2_mean': np.mean(config2_results)
                }
    
    results['config_comparison_tests'] = config_comparison_tests
    
    # Analyze the impact of captures on game outcome
    capture_impact_tests = {}
    
    # Compare win rates between games with different number of captures
    for captures in range(5):
        capture_games = df[df['goats_captured'] == captures]
        no_capture_games = df[df['goats_captured'] == 0]
        
        if len(capture_games) > 0 and len(no_capture_games) > 0:
            # Calculate adjusted win rates for Tiger
            capture_results = capture_games['tiger_won'].astype(float) + 0.5 * capture_games['draw'].astype(float)
            no_capture_results = no_capture_games['tiger_won'].astype(float) + 0.5 * no_capture_games['draw'].astype(float)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(capture_results, no_capture_results, equal_var=False)
            
            capture_impact_tests[f"captures_{captures}_vs_0"] = {
                'statistic': t_stat,
                'p_value': p_value,
                'captures_mean': np.mean(capture_results),
                'no_captures_mean': np.mean(no_capture_results)
            }
    
    results['capture_impact_tests'] = capture_impact_tests
    
    return results 

def generate_statistical_report(statistical_results, output_dir, config=None):
    """
    Generate a dedicated statistical report with test results.
    
    Args:
        statistical_results: Dictionary with statistical test results
        output_dir: Directory to save the report
        config: Configuration dictionary
    """
    report_path = os.path.join(output_dir, 'statistical_validation.txt')
    significance_threshold = config.get('statistical', {}).get('significance_threshold', 0.05)
    
    with open(report_path, 'w') as f:
        f.write("STATISTICAL VALIDATION REPORT\n")
        f.write("============================\n\n")
        
        f.write(f"Significance level (α): {significance_threshold}\n")
        f.write("Multiple comparison correction: Benjamini-Hochberg procedure\n\n")
        
        # Algorithm comparison
        if 'algorithm_comparison_test' in statistical_results:
            test = statistical_results['algorithm_comparison_test']
            f.write("ALGORITHM COMPARISON (MCTS vs Minimax)\n")
            f.write("---------------------------------\n")
            f.write(f"Test: Two-sample t-test (Welch's t-test with unequal variance)\n")
            f.write(f"t-statistic: {test['statistic']:.4f}\n")
            f.write(f"p-value: {test['p_value']:.4f}\n")
            f.write(f"Significant: {'Yes' if test['p_value'] < significance_threshold else 'No'}\n\n")
            
            # Add interpretation
            if test['p_value'] < significance_threshold:
                better_algo = "MCTS" if test['mcts_mean'] > test['minimax_mean'] else "Minimax"
                f.write(f"Interpretation: {better_algo} performs significantly better overall.\n")
                effect_size = abs(test['mcts_mean'] - test['minimax_mean'])
                effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                f.write(f"  MCTS win rate: {test['mcts_mean']:.4f}\n")
                f.write(f"  Minimax win rate: {test['minimax_mean']:.4f}\n")
                f.write(f"  Effect size (mean difference): {effect_size:.4f} ({effect_magnitude})\n")
            else:
                f.write("Interpretation: No significant difference in overall performance.\n")
            f.write("\n")
        
        # Depth comparisons with multiple comparison correction
        if 'depth_comparison_tests' in statistical_results and statistical_results['depth_comparison_tests']:
            f.write("MINIMAX DEPTH COMPARISONS\n")
            f.write("-------------------------\n")
            f.write("Test: Pairwise t-tests with Benjamini-Hochberg correction\n\n")
            
            # Get p-values for correction
            tests = statistical_results['depth_comparison_tests']
            test_keys = list(tests.keys())
            all_p_values = [test['p_value'] for test in tests.values()]
            
            # Apply correction if we have p-values
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                # Report results
                for i, key in enumerate(test_keys):
                    test = tests[key]
                    d1, d2 = key.split('_vs_')
                    corrected_p = corrected_p_values[i]
                    is_significant = corrected_p < significance_threshold
                    
                    f.write(f"Depth {d1} vs Depth {d2}:\n")
                    f.write(f"  t-statistic: {test['statistic']:.4f}\n")
                    f.write(f"  Original p-value: {test['p_value']:.4f}\n")
                    f.write(f"  Corrected p-value: {corrected_p:.4f}\n")
                    f.write(f"  Significant after correction: {'Yes' if is_significant else 'No'}\n")
                    
                    if 'depth1_mean' in test and 'depth2_mean' in test:
                        better_depth = d1 if test['depth1_mean'] > test['depth2_mean'] else d2
                        effect_size = abs(test['depth1_mean'] - test['depth2_mean'])
                        effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                        f.write(f"  Better performer: Depth {better_depth}\n")
                        f.write(f"  Effect size (mean difference): {effect_size:.4f} ({effect_magnitude})\n")
                    f.write("\n")
        
        # MCTS configuration comparisons with multiple comparison correction
        if 'config_comparison_tests' in statistical_results and statistical_results['config_comparison_tests']:
            f.write("MCTS CONFIGURATION COMPARISONS\n")
            f.write("-----------------------------\n")
            f.write("Test: Pairwise t-tests with Benjamini-Hochberg correction\n\n")
            
            # Get p-values for correction
            tests = statistical_results['config_comparison_tests']
            test_keys = list(tests.keys())
            all_p_values = [test['p_value'] for test in tests.values()]
            
            # Apply correction if we have p-values
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                # Find significant comparisons
                significant_tests = []
                for i, key in enumerate(test_keys):
                    if corrected_p_values[i] < significance_threshold:
                        significant_tests.append((key, tests[key], corrected_p_values[i]))
                
                if significant_tests:
                    f.write(f"Found {len(significant_tests)} significant configuration differences after correction:\n\n")
                    
                    for key, test, corrected_p in significant_tests:
                        config1, config2 = key.split('_vs_')
                        f.write(f"{config1} vs {config2}:\n")
                        f.write(f"  t-statistic: {test['statistic']:.4f}\n")
                        f.write(f"  Original p-value: {test['p_value']:.4f}\n")
                        f.write(f"  Corrected p-value: {corrected_p:.4f}\n")
                        
                        if 'config1_mean' in test and 'config2_mean' in test:
                            better_config = config1 if test['config1_mean'] > test['config2_mean'] else config2
                            effect_size = abs(test['config1_mean'] - test['config2_mean'])
                            effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                            f.write(f"  Better configuration: {better_config}\n")
                            f.write(f"  Effect size (mean difference): {effect_size:.4f} ({effect_magnitude})\n")
                        f.write("\n")
                else:
                    f.write("No significant differences between MCTS configurations after correction for multiple comparisons.\n\n")
        
        # Capture impact tests
        if 'capture_impact_tests' in statistical_results and statistical_results['capture_impact_tests']:
            f.write("CAPTURE IMPACT ANALYSIS\n")
            f.write("---------------------\n")
            f.write("Test: t-tests comparing games with captures vs no captures\n\n")
            
            # Get p-values for correction
            tests = statistical_results['capture_impact_tests']
            test_keys = list(tests.keys())
            all_p_values = [test['p_value'] for test in tests.values()]
            
            # Apply correction if we have p-values
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                # Report results
                for i, key in enumerate(test_keys):
                    test = tests[key]
                    captures = key.split('_vs_')[0].replace('captures_', '')
                    corrected_p = corrected_p_values[i]
                    is_significant = corrected_p < significance_threshold
                    
                    f.write(f"{captures} captures vs 0 captures:\n")
                    f.write(f"  t-statistic: {test['statistic']:.4f}\n")
                    f.write(f"  Original p-value: {test['p_value']:.4f}\n")
                    f.write(f"  Corrected p-value: {corrected_p:.4f}\n")
                    f.write(f"  Significant after correction: {'Yes' if is_significant else 'No'}\n")
                    
                    if 'captures_mean' in test and 'no_captures_mean' in test:
                        effect_size = abs(test['captures_mean'] - test['no_captures_mean'])
                        effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                        f.write(f"  Effect size (mean difference): {effect_size:.4f} ({effect_magnitude})\n")
                        
                        # Determine which side is advantaged
                        if is_significant:
                            advantaged = "Tigers" if test['captures_mean'] > test['no_captures_mean'] else "Goats"
                            f.write(f"  Interpretation: {advantaged} are significantly advantaged in games with {captures} captures\n")
                    f.write("\n")
        
        f.write("\nASSUMPTION CHECKS\n")
        f.write("----------------\n")
        f.write("Note: For large sample sizes (n > 30), parametric tests are generally robust\n")
        f.write("to violations of normality due to the Central Limit Theorem.\n\n")
        
        if 'algorithm_comparison_test' in statistical_results:
            f.write("Sample sizes for algorithm comparison:\n")
            if 'sample_sizes' in statistical_results.get('assumption_checks', {}):
                sample_sizes = statistical_results['assumption_checks']['sample_sizes']
                f.write(f"  MCTS: {sample_sizes[0]} games\n")
                f.write(f"  Minimax: {sample_sizes[1]} games\n")
            else:
                f.write("  Sample size information not available\n")
            f.write("\n")
            
        f.write("\nSTATISTICALLY SIGNIFICANT FINDINGS SUMMARY\n")
        f.write("---------------------------------------\n")
        
        # Collect all significant findings after correction
        significant_findings = []
        
        # Algorithm comparison
        if 'algorithm_comparison_test' in statistical_results:
            test = statistical_results['algorithm_comparison_test']
            if test['p_value'] < significance_threshold:
                better_algo = "MCTS" if test['mcts_mean'] > test['minimax_mean'] else "Minimax"
                significant_findings.append(f"- {better_algo} performs significantly better overall (p={test['p_value']:.4f})")
        
        # Depth comparisons
        if 'depth_comparison_tests' in statistical_results and statistical_results['depth_comparison_tests']:
            tests = statistical_results['depth_comparison_tests']
            all_p_values = [test['p_value'] for test in tests.values()]
            
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                for i, (key, test) in enumerate(tests.items()):
                    if corrected_p_values[i] < significance_threshold:
                        d1, d2 = key.split('_vs_')
                        better_depth = d1 if test.get('depth1_mean', 0) > test.get('depth2_mean', 0) else d2
                        significant_findings.append(f"- Minimax depth {better_depth} performs significantly better than depth {d2 if better_depth == d1 else d1} (corrected p={corrected_p_values[i]:.4f})")
        
        # MCTS configuration comparisons
        if 'config_comparison_tests' in statistical_results and statistical_results['config_comparison_tests']:
            tests = statistical_results['config_comparison_tests']
            all_p_values = [test['p_value'] for test in tests.values()]
            
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                for i, (key, test) in enumerate(tests.items()):
                    if corrected_p_values[i] < significance_threshold:
                        config1, config2 = key.split('_vs_')
                        better_config = config1 if test.get('config1_mean', 0) > test.get('config2_mean', 0) else config2
                        significant_findings.append(f"- MCTS configuration {better_config} performs significantly better than {config2 if better_config == config1 else config1} (corrected p={corrected_p_values[i]:.4f})")
        
        # Capture impact tests
        if 'capture_impact_tests' in statistical_results and statistical_results['capture_impact_tests']:
            tests = statistical_results['capture_impact_tests']
            all_p_values = [test['p_value'] for test in tests.values()]
            
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                
                for i, (key, test) in enumerate(tests.items()):
                    if corrected_p_values[i] < significance_threshold:
                        captures = key.split('_vs_')[0].replace('captures_', '')
                        advantaged = "Tigers" if test.get('captures_mean', 0) > test.get('no_captures_mean', 0) else "Goats"
                        significant_findings.append(f"- {advantaged} are significantly advantaged in games with {captures} captures (corrected p={corrected_p_values[i]:.4f})")
        
        if significant_findings:
            for finding in significant_findings:
                f.write(f"{finding}\n")
        else:
            f.write("No findings remained statistically significant after correction for multiple comparisons.\n")
        
    print(f"Statistical validation report saved to {report_path}")
    return report_path 