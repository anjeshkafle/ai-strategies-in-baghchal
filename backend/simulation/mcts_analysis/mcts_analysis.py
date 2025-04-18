"""
Core analysis functions for MCTS tournament results.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import random

def load_and_preprocess_data(filepath):
    """
    Load tournament results and preprocess for analysis.
    - Balance tiger/goat representation
    - Extract configuration parameters
    - Calculate basic metrics
    
    Args:
        filepath: Path to the tournament results CSV file
        
    Returns:
        DataFrame with preprocessed data
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Extract configuration parameters
    df['tiger_config_parsed'] = df['tiger_config'].apply(lambda x: json.loads(x))
    df['goat_config_parsed'] = df['goat_config'].apply(lambda x: json.loads(x))
    
    # Extract parameters for tiger
    df['tiger_rollout_policy'] = df['tiger_config_parsed'].apply(lambda x: x.get('rollout_policy'))
    df['tiger_rollout_depth'] = df['tiger_config_parsed'].apply(lambda x: x.get('rollout_depth'))
    df['tiger_exploration_weight'] = df['tiger_config_parsed'].apply(lambda x: x.get('exploration_weight'))
    df['tiger_guided_strictness'] = df['tiger_config_parsed'].apply(lambda x: x.get('guided_strictness'))
    
    # Extract parameters for goat
    df['goat_rollout_policy'] = df['goat_config_parsed'].apply(lambda x: x.get('rollout_policy'))
    df['goat_rollout_depth'] = df['goat_config_parsed'].apply(lambda x: x.get('rollout_depth'))
    df['goat_exploration_weight'] = df['goat_config_parsed'].apply(lambda x: x.get('exploration_weight'))
    df['goat_guided_strictness'] = df['goat_config_parsed'].apply(lambda x: x.get('guided_strictness'))
    
    # Create identifier for each unique configuration
    df['tiger_config_id'] = df.apply(
        lambda row: f"mcts_{row['tiger_rollout_policy']}_{row['tiger_rollout_depth']}_{row['tiger_exploration_weight']}",
        axis=1
    )
    df['goat_config_id'] = df.apply(
        lambda row: f"mcts_{row['goat_rollout_policy']}_{row['goat_rollout_depth']}_{row['goat_exploration_weight']}",
        axis=1
    )
    
    # Add outcome columns for each configuration
    df['tiger_won'] = df['winner'] == 'TIGER'
    df['goat_won'] = df['winner'] == 'GOAT'
    df['draw'] = df['winner'] == 'DRAW'
    
    # Calculate game statistics
    df['game_length'] = df['moves'].astype(int)
    df['first_capture_move'] = df['first_capture_move'].fillna(0).astype(int)
    df['goats_captured'] = df['goats_captured'].fillna(0).astype(int)
    
    return df

def calculate_win_rates(df):
    """
    Calculate win rates per configuration, overall and by side.
    
    Args:
        df: Preprocessed tournament data
        
    Returns:
        DataFrame with win rates for each configuration
    """
    # Calculate win rates when playing as tiger
    tiger_stats = []
    for config in df['tiger_config_id'].unique():
        tiger_games = df[df['tiger_config_id'] == config]
        if len(tiger_games) == 0:
            continue
            
        stats = {
            'config_id': config,
            'total_games_as_tiger': len(tiger_games),
            'wins_as_tiger': tiger_games['tiger_won'].sum(),
            'draws_as_tiger': tiger_games['draw'].sum(),
            'losses_as_tiger': tiger_games['goat_won'].sum(),
            'win_rate_as_tiger': tiger_games['tiger_won'].sum() / len(tiger_games),
            'draw_rate_as_tiger': tiger_games['draw'].sum() / len(tiger_games),
            'avg_game_length_as_tiger': tiger_games['game_length'].mean(),
            'avg_goats_captured': tiger_games['goats_captured'].mean()
        }
        
        # Extract parameters
        params = tiger_games.iloc[0]
        stats['rollout_policy'] = params['tiger_rollout_policy']
        stats['rollout_depth'] = params['tiger_rollout_depth']
        stats['exploration_weight'] = params['tiger_exploration_weight']
        stats['guided_strictness'] = params['tiger_guided_strictness'] if params['tiger_rollout_policy'] == 'guided' else None
        
        tiger_stats.append(stats)
    
    tiger_df = pd.DataFrame(tiger_stats)
    
    # Calculate win rates when playing as goat
    goat_stats = []
    for config in df['goat_config_id'].unique():
        goat_games = df[df['goat_config_id'] == config]
        if len(goat_games) == 0:
            continue
            
        stats = {
            'config_id': config,
            'total_games_as_goat': len(goat_games),
            'wins_as_goat': goat_games['goat_won'].sum(),
            'draws_as_goat': goat_games['draw'].sum(),
            'losses_as_goat': goat_games['tiger_won'].sum(),
            'win_rate_as_goat': goat_games['goat_won'].sum() / len(goat_games),
            'draw_rate_as_goat': goat_games['draw'].sum() / len(goat_games),
            'avg_game_length_as_goat': goat_games['game_length'].mean()
        }
        
        # Extract parameters
        params = goat_games.iloc[0]
        stats['rollout_policy'] = params['goat_rollout_policy']
        stats['rollout_depth'] = params['goat_rollout_depth']
        stats['exploration_weight'] = params['goat_exploration_weight']
        stats['guided_strictness'] = params['goat_guided_strictness'] if params['goat_rollout_policy'] == 'guided' else None
        
        goat_stats.append(stats)
    
    goat_df = pd.DataFrame(goat_stats)
    
    # Merge tiger and goat stats
    merged_df = pd.merge(
        tiger_df, 
        goat_df, 
        on=['config_id', 'rollout_policy', 'rollout_depth', 'exploration_weight', 'guided_strictness'],
        how='outer'
    ).fillna(0)
    
    # Calculate overall metrics
    merged_df['total_games'] = merged_df['total_games_as_tiger'] + merged_df['total_games_as_goat']
    merged_df['total_wins'] = merged_df['wins_as_tiger'] + merged_df['wins_as_goat']
    merged_df['total_draws'] = merged_df['draws_as_tiger'] + merged_df['draws_as_goat']
    merged_df['total_losses'] = merged_df['losses_as_tiger'] + merged_df['losses_as_goat']
    
    # Calculate overall win rate (weighted average of tiger and goat performance)
    tiger_weight = merged_df['total_games_as_tiger'] / merged_df['total_games']
    goat_weight = merged_df['total_games_as_goat'] / merged_df['total_games']
    
    merged_df['overall_win_rate'] = (
        merged_df['win_rate_as_tiger'] * tiger_weight + 
        merged_df['win_rate_as_goat'] * goat_weight
    )
    
    merged_df['average_win_rate'] = merged_df['total_wins'] / merged_df['total_games']
    
    # Calculate adjusted win rates that count draws as 0.5 points
    merged_df['adjusted_win_rate_as_tiger'] = (merged_df['wins_as_tiger'] + 0.5 * merged_df['draws_as_tiger']) / merged_df['total_games_as_tiger']
    merged_df['adjusted_win_rate_as_goat'] = (merged_df['wins_as_goat'] + 0.5 * merged_df['draws_as_goat']) / merged_df['total_games_as_goat']
    
    # Calculate overall adjusted win rate that counts draws as 0.5 points
    merged_df['adjusted_win_rate'] = (merged_df['total_wins'] + 0.5 * merged_df['total_draws']) / merged_df['total_games']
    
    # Sort by adjusted win rate instead of average win rate to better reflect performance
    merged_df = merged_df.sort_values('adjusted_win_rate', ascending=False)
    
    return merged_df

def perform_statistical_tests(df):
    """
    Perform t-tests and ANOVA on configuration parameters.
    
    Args:
        df: Preprocessed tournament data
        
    Returns:
        Dictionary with statistical test results
    """
    results = {}
    
    # T-test for rollout depth (4 vs 6)
    depth4_games = df[(df['tiger_rollout_depth'] == 4) | (df['goat_rollout_depth'] == 4)]
    depth6_games = df[(df['tiger_rollout_depth'] == 6) | (df['goat_rollout_depth'] == 6)]
    
    depth4_win_rate = depth4_games['tiger_won'].mean() if 'tiger_won' in depth4_games else 0
    depth6_win_rate = depth6_games['tiger_won'].mean() if 'tiger_won' in depth6_games else 0
    
    depth_ttest = stats.ttest_ind(
        depth4_games['game_length'].dropna(),
        depth6_games['game_length'].dropna(),
        equal_var=False
    )
    
    results['depth_ttest'] = {
        'statistic': depth_ttest.statistic,
        'p_value': depth_ttest.pvalue,
        'depth4_win_rate': depth4_win_rate,
        'depth6_win_rate': depth6_win_rate
    }
    
    # T-tests for exploration weights
    weights = [1.0, 1.414, 2.0]
    weight_results = {}
    
    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            if i >= j:
                continue
                
            w1_games = df[(df['tiger_exploration_weight'] == w1) | (df['goat_exploration_weight'] == w1)]
            w2_games = df[(df['tiger_exploration_weight'] == w2) | (df['goat_exploration_weight'] == w2)]
            
            if len(w1_games) == 0 or len(w2_games) == 0:
                continue
                
            weight_ttest = stats.ttest_ind(
                w1_games['game_length'].dropna(),
                w2_games['game_length'].dropna(),
                equal_var=False
            )
            
            weight_results[f'{w1}_vs_{w2}'] = {
                'statistic': weight_ttest.statistic,
                'p_value': weight_ttest.pvalue
            }
    
    results['exploration_weight_ttests'] = weight_results
    
    # ANOVA for rollout policies
    policy_groups = []
    policy_names = []
    
    for policy in ['random', 'lightweight', 'guided']:
        policy_games = df[(df['tiger_rollout_policy'] == policy) | (df['goat_rollout_policy'] == policy)]
        if len(policy_games) > 0:
            policy_groups.append(policy_games['game_length'].dropna())
            policy_names.append(policy)
    
    if len(policy_groups) > 1:
        policy_anova = stats.f_oneway(*policy_groups)
        
        results['policy_anova'] = {
            'statistic': policy_anova.statistic,
            'p_value': policy_anova.pvalue,
            'policies': policy_names
        }
    
    return results

def calculate_elo_ratings(df, K=32, initial_rating=1500):
    """
    Calculate Elo ratings for each configuration.
    
    Args:
        df: Preprocessed tournament data
        K: K-factor for Elo calculation
        initial_rating: Initial rating for all configurations
        
    Returns:
        DataFrame with Elo ratings for each configuration
    """
    # Get all unique configurations
    all_configs = set(df['tiger_config_id'].unique()) | set(df['goat_config_id'].unique())
    
    # Initialize ratings
    ratings = {config: initial_rating for config in all_configs}
    
    # Function to calculate expected score
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Function to update ratings
    def update_rating(rating, expected, actual, k_factor):
        return rating + k_factor * (actual - expected)
    
    # Process each game
    for _, game in df.iterrows():
        tiger_config = game['tiger_config_id']
        goat_config = game['goat_config_id']
        
        # Skip if configurations are not tracked
        if tiger_config not in ratings or goat_config not in ratings:
            continue
        
        tiger_rating = ratings[tiger_config]
        goat_rating = ratings[goat_config]
        
        # Calculate expected scores
        tiger_expected = expected_score(tiger_rating, goat_rating)
        goat_expected = expected_score(goat_rating, tiger_rating)
        
        # Determine actual scores
        if game['tiger_won']:
            tiger_actual = 1.0
            goat_actual = 0.0
        elif game['goat_won']:
            tiger_actual = 0.0
            goat_actual = 1.0
        else:  # Draw
            tiger_actual = 0.5
            goat_actual = 0.5
        
        # Update ratings
        ratings[tiger_config] = update_rating(tiger_rating, tiger_expected, tiger_actual, K)
        ratings[goat_config] = update_rating(goat_rating, goat_expected, goat_actual, K)
    
    # Prepare results DataFrame
    elo_results = []
    
    for config in all_configs:
        if config in ratings:
            # Extract parameters from config ID
            parts = config.split('_')
            if len(parts) >= 4:
                rollout_policy = parts[1]
                rollout_depth = int(parts[2]) if parts[2].isdigit() else None
                exploration_weight = float(parts[3]) if parts[3] and parts[3] != 'None' else None
                
                elo_results.append({
                    'config_id': config,
                    'elo_rating': ratings[config],
                    'rollout_policy': rollout_policy,
                    'rollout_depth': rollout_depth,
                    'exploration_weight': exploration_weight
                })
    
    elo_df = pd.DataFrame(elo_results)
    elo_df = elo_df.sort_values('elo_rating', ascending=False)
    
    return elo_df

def generate_composite_scores(win_rates_df, elo_df):
    """
    Generate composite scores from win rates and Elo ratings.
    
    Args:
        win_rates_df: DataFrame with win rates
        elo_df: DataFrame with Elo ratings
        
    Returns:
        DataFrame with composite scores
    """
    # Merge win rates and Elo ratings
    merged_df = pd.merge(
        win_rates_df,
        elo_df[['config_id', 'elo_rating']],
        on='config_id',
        how='inner'
    )
    
    # Normalize Elo ratings to 0-1 scale
    min_elo = merged_df['elo_rating'].min()
    max_elo = merged_df['elo_rating'].max()
    merged_df['normalized_elo'] = (merged_df['elo_rating'] - min_elo) / (max_elo - min_elo) if max_elo > min_elo else 0.5
    
    # Calculate composite score (50% adjusted win rate, 50% Elo)
    merged_df['composite_score'] = 0.5 * merged_df['adjusted_win_rate'] + 0.5 * merged_df['normalized_elo']
    
    # Sort by composite score
    merged_df = merged_df.sort_values('composite_score', ascending=False)
    
    return merged_df

def select_top_configurations(composite_scores_df, n=3):
    """
    Select top n configurations considering diversity.
    
    Args:
        composite_scores_df: DataFrame with composite scores
        n: Number of configurations to select
        
    Returns:
        List of dictionaries representing top configurations
    """
    # Start with top configuration
    top_configs = []
    
    # Try to ensure diverse configurations (different policies and depths)
    selected_policies = set()
    selected_depths = set()
    
    # Select configurations in order, but promote diversity
    candidates = composite_scores_df.copy()
    
    while len(top_configs) < n and not candidates.empty:
        # Get the top candidate
        top_candidate = candidates.iloc[0]
        
        # Extract configuration details
        config = {
            'config_id': top_candidate['config_id'],
            'rollout_policy': top_candidate['rollout_policy'],
            'rollout_depth': int(top_candidate['rollout_depth']),
            'exploration_weight': float(top_candidate['exploration_weight']),
            'composite_score': float(top_candidate['composite_score']),
            'adjusted_win_rate': float(top_candidate['adjusted_win_rate']),
            'average_win_rate': float(top_candidate['average_win_rate']),
            'elo_rating': float(top_candidate['elo_rating'])
        }
        
        # If we already have n-1 configurations, just add the best remaining one
        if len(top_configs) == n - 1:
            top_configs.append(config)
            break
            
        # Otherwise, check for diversity
        if (top_candidate['rollout_policy'] not in selected_policies or 
            top_candidate['rollout_depth'] not in selected_depths or
            len(top_configs) < 2):  # Always include the top 2 regardless of diversity
            
            top_configs.append(config)
            selected_policies.add(top_candidate['rollout_policy'])
            selected_depths.add(top_candidate['rollout_depth'])
            
        # Remove this candidate from further consideration
        candidates = candidates.iloc[1:]
        
    # If we still need more configurations, just take the top remaining ones
    while len(top_configs) < n and not candidates.empty:
        top_candidate = candidates.iloc[0]
        
        config = {
            'config_id': top_candidate['config_id'],
            'rollout_policy': top_candidate['rollout_policy'],
            'rollout_depth': int(top_candidate['rollout_depth']),
            'exploration_weight': float(top_candidate['exploration_weight']),
            'composite_score': float(top_candidate['composite_score']),
            'adjusted_win_rate': float(top_candidate['adjusted_win_rate']),
            'average_win_rate': float(top_candidate['average_win_rate']),
            'elo_rating': float(top_candidate['elo_rating'])
        }
        
        top_configs.append(config)
        candidates = candidates.iloc[1:]
    
    return top_configs 