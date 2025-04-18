"""
Core analysis functions for MCTS tournament results.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import random

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
        "elo": {
            "initial_rating": 1500,
            "base_k_factor": 32
        },
        "composite_score": {
            "win_rate_weight": 0.5,
            "elo_weight": 0.5
        },
        "top_configs": {
            "count": 3,
            "enforce_diversity": True
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
    from scipy import stats
    
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
    
    # Extract configuration parameters with error handling
    def safe_json_load(json_str):
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"ERROR parsing JSON: {e}")
            print(f"Problematic JSON string: {json_str}")
            return {}
    
    df['tiger_config_parsed'] = df['tiger_config'].apply(safe_json_load)
    df['goat_config_parsed'] = df['goat_config'].apply(safe_json_load)
    
    # Extract parameters for tiger with more detailed error trapping
    def safe_get_param(config_dict, param_name, default_value=None):
        try:
            return config_dict.get(param_name, default_value)
        except Exception as e:
            print(f"ERROR extracting {param_name}: {e}")
            return default_value
    
    df['tiger_rollout_policy'] = df['tiger_config_parsed'].apply(lambda x: safe_get_param(x, 'rollout_policy'))
    df['tiger_rollout_depth'] = df['tiger_config_parsed'].apply(lambda x: safe_get_param(x, 'rollout_depth'))
    df['tiger_exploration_weight'] = df['tiger_config_parsed'].apply(lambda x: safe_get_param(x, 'exploration_weight'))
    df['tiger_guided_strictness'] = df['tiger_config_parsed'].apply(lambda x: safe_get_param(x, 'guided_strictness'))
    
    # Extract parameters for goat
    df['goat_rollout_policy'] = df['goat_config_parsed'].apply(lambda x: safe_get_param(x, 'rollout_policy'))
    df['goat_rollout_depth'] = df['goat_config_parsed'].apply(lambda x: safe_get_param(x, 'rollout_depth'))
    df['goat_exploration_weight'] = df['goat_config_parsed'].apply(lambda x: safe_get_param(x, 'exploration_weight'))
    df['goat_guided_strictness'] = df['goat_config_parsed'].apply(lambda x: safe_get_param(x, 'guided_strictness'))
    
    # Create identifier for each unique configuration
    df['tiger_config_id'] = df.apply(
        lambda row: f"mcts_{row['tiger_rollout_policy']}_{row['tiger_rollout_depth']}_{row['tiger_exploration_weight']}" + 
                   (f"_{row['tiger_guided_strictness']}" if row['tiger_rollout_policy'] == 'guided' and pd.notna(row['tiger_guided_strictness']) else ""),
        axis=1
    )
    df['goat_config_id'] = df.apply(
        lambda row: f"mcts_{row['goat_rollout_policy']}_{row['goat_rollout_depth']}_{row['goat_exploration_weight']}" + 
                   (f"_{row['goat_guided_strictness']}" if row['goat_rollout_policy'] == 'guided' and pd.notna(row['goat_guided_strictness']) else ""),
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
    # Get all unique configurations
    all_config_ids = set(df['tiger_config_id'].unique()) | set(df['goat_config_id'].unique())
    
    # Calculate win rates when playing as tiger
    tiger_stats = []
    
    for config in df['tiger_config_id'].unique():
        tiger_games = df[df['tiger_config_id'] == config]
        if len(tiger_games) == 0:
            continue
            
        tiger_win_rate = tiger_games['tiger_won'].sum() / len(tiger_games)
        tiger_draw_rate = tiger_games['draw'].sum() / len(tiger_games)
        
        stats = {
            'config_id': config,
            'total_games_as_tiger': len(tiger_games),
            'wins_as_tiger': tiger_games['tiger_won'].sum(),
            'draws_as_tiger': tiger_games['draw'].sum(),
            'losses_as_tiger': tiger_games['goat_won'].sum(),
            'win_rate_as_tiger': tiger_win_rate,
            'draw_rate_as_tiger': tiger_draw_rate,
            'avg_game_length_as_tiger': tiger_games['game_length'].mean(),
            'avg_goats_captured': tiger_games['goats_captured'].mean()
        }
        
        # Add confidence intervals
        stats['tiger_ci_lower'], stats['tiger_ci_upper'] = calculate_confidence_intervals(
            tiger_win_rate, 
            len(tiger_games)
        )
        
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
        
        goat_win_rate = goat_games['goat_won'].sum() / len(goat_games)
        goat_draw_rate = goat_games['draw'].sum() / len(goat_games)
            
        stats = {
            'config_id': config,
            'total_games_as_goat': len(goat_games),
            'wins_as_goat': goat_games['goat_won'].sum(),
            'draws_as_goat': goat_games['draw'].sum(),
            'losses_as_goat': goat_games['tiger_won'].sum(),
            'win_rate_as_goat': goat_win_rate,
            'draw_rate_as_goat': goat_draw_rate,
            'avg_game_length_as_goat': goat_games['game_length'].mean()
        }
        
        # Add confidence intervals
        stats['goat_ci_lower'], stats['goat_ci_upper'] = calculate_confidence_intervals(
            goat_win_rate, 
            len(goat_games)
        )
        
        # Extract parameters
        params = goat_games.iloc[0]
        stats['rollout_policy'] = params['goat_rollout_policy']
        stats['rollout_depth'] = params['goat_rollout_depth']
        stats['exploration_weight'] = params['goat_exploration_weight']
        stats['guided_strictness'] = params['goat_guided_strictness'] if params['goat_rollout_policy'] == 'guided' else None
        
        goat_stats.append(stats)
    
    goat_df = pd.DataFrame(goat_stats)
    
    # First create a dataframe with all unique config_ids
    all_configs_df = pd.DataFrame({'config_id': list(all_config_ids)})
    
    # Merge tiger stats
    merged_df = pd.merge(
        all_configs_df,
        tiger_df,
        on='config_id',
        how='left'
    )
    
    # Merge goat stats
    merged_df = pd.merge(
        merged_df,
        goat_df,
        on='config_id',
        how='left',
        suffixes=('', '_goat')
    )
    
    # Rename columns from goat dataframe that didn't get a suffix
    rename_cols = {}
    for col in goat_df.columns:
        if col != 'config_id' and col in merged_df.columns and f"{col}_goat" in merged_df.columns:
            continue  # Already has suffix
        elif col != 'config_id':
            rename_cols[col] = f"{col}_goat"
    
    if rename_cols:
        merged_df = merged_df.rename(columns=rename_cols)
    
    # Fill missing values
    merged_df = merged_df.fillna(0)
    
    # Handle parameter columns
    for param in ['rollout_policy', 'rollout_depth', 'exploration_weight', 'guided_strictness']:
        param_goat = f"{param}_goat"
        if param_goat in merged_df.columns:
            # Use tiger param if available, otherwise use goat param
            merged_df[param] = merged_df.apply(
                lambda row: row[param] if row[param] != 0 else row[param_goat], 
                axis=1
            )
    
    # Fix column names for calculations
    column_mapping = {
        'total_games_as_tiger': 'total_games_as_tiger',
        'wins_as_tiger': 'wins_as_tiger',
        'draws_as_tiger': 'draws_as_tiger',
        'losses_as_tiger': 'losses_as_tiger',
        'win_rate_as_tiger': 'win_rate_as_tiger',
        'draw_rate_as_tiger': 'draw_rate_as_tiger',
        'avg_game_length_as_tiger': 'avg_game_length_as_tiger',
        'avg_goats_captured': 'avg_goats_captured',
        'tiger_ci_lower': 'tiger_ci_lower',
        'tiger_ci_upper': 'tiger_ci_upper',
        
        'total_games_as_goat': 'total_games_as_goat_goat' if 'total_games_as_goat_goat' in merged_df.columns else 'total_games_as_goat',
        'wins_as_goat': 'wins_as_goat_goat' if 'wins_as_goat_goat' in merged_df.columns else 'wins_as_goat',
        'draws_as_goat': 'draws_as_goat_goat' if 'draws_as_goat_goat' in merged_df.columns else 'draws_as_goat',
        'losses_as_goat': 'losses_as_goat_goat' if 'losses_as_goat_goat' in merged_df.columns else 'losses_as_goat',
        'win_rate_as_goat': 'win_rate_as_goat_goat' if 'win_rate_as_goat_goat' in merged_df.columns else 'win_rate_as_goat',
        'draw_rate_as_goat': 'draw_rate_as_goat_goat' if 'draw_rate_as_goat_goat' in merged_df.columns else 'draw_rate_as_goat',
        'avg_game_length_as_goat': 'avg_game_length_as_goat_goat' if 'avg_game_length_as_goat_goat' in merged_df.columns else 'avg_game_length_as_goat',
        'goat_ci_lower': 'goat_ci_lower_goat' if 'goat_ci_lower_goat' in merged_df.columns else 'goat_ci_lower',
        'goat_ci_upper': 'goat_ci_upper_goat' if 'goat_ci_upper_goat' in merged_df.columns else 'goat_ci_upper'
    }
    
    # Calculate overall metrics using the corrected column names
    merged_df['total_games'] = merged_df[column_mapping['total_games_as_tiger']] + merged_df[column_mapping['total_games_as_goat']]
    merged_df['total_wins'] = merged_df[column_mapping['wins_as_tiger']] + merged_df[column_mapping['wins_as_goat']]
    merged_df['total_draws'] = merged_df[column_mapping['draws_as_tiger']] + merged_df[column_mapping['draws_as_goat']]
    merged_df['total_losses'] = merged_df[column_mapping['losses_as_tiger']] + merged_df[column_mapping['losses_as_goat']]
    
    # Calculate overall win rate (weighted average of tiger and goat performance)
    tiger_weight = merged_df[column_mapping['total_games_as_tiger']] / merged_df['total_games']
    goat_weight = merged_df[column_mapping['total_games_as_goat']] / merged_df['total_games']
    
    merged_df['overall_win_rate'] = (
        merged_df[column_mapping['win_rate_as_tiger']] * tiger_weight + 
        merged_df[column_mapping['win_rate_as_goat']] * goat_weight
    )
    
    merged_df['average_win_rate'] = merged_df['total_wins'] / merged_df['total_games']
    
    # Calculate adjusted win rates that count draws as 0.5 points
    merged_df['adjusted_win_rate_as_tiger'] = (merged_df[column_mapping['wins_as_tiger']] + 0.5 * merged_df[column_mapping['draws_as_tiger']]) / merged_df[column_mapping['total_games_as_tiger']]
    merged_df['adjusted_win_rate_as_goat'] = (merged_df[column_mapping['wins_as_goat']] + 0.5 * merged_df[column_mapping['draws_as_goat']]) / merged_df[column_mapping['total_games_as_goat']]
    
    # Calculate overall adjusted win rate that counts draws as 0.5 points
    merged_df['adjusted_win_rate'] = (merged_df['total_wins'] + 0.5 * merged_df['total_draws']) / merged_df['total_games']
    
    # Calculate confidence intervals for adjusted win rates
    merged_df['adjusted_win_rate_ci_lower'], merged_df['adjusted_win_rate_ci_upper'] = zip(
        *merged_df.apply(
            lambda row: calculate_confidence_intervals(row['adjusted_win_rate'], row['total_games']),
            axis=1
        )
    )
    
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
    
    # Compare win rates instead of game lengths
    depth4_win_scores = []
    for _, game in depth4_games.iterrows():
        if game['tiger_rollout_depth'] == 4:
            score = 1 if game['tiger_won'] else 0.5 if game['draw'] else 0
        elif game['goat_rollout_depth'] == 4:
            score = 1 if game['goat_won'] else 0.5 if game['draw'] else 0
        depth4_win_scores.append(score)
    
    depth6_win_scores = []
    for _, game in depth6_games.iterrows():
        if game['tiger_rollout_depth'] == 6:
            score = 1 if game['tiger_won'] else 0.5 if game['draw'] else 0
        elif game['goat_rollout_depth'] == 6:
            score = 1 if game['goat_won'] else 0.5 if game['draw'] else 0
        depth6_win_scores.append(score)
    
    if depth4_win_scores and depth6_win_scores:
        depth_ttest = stats.ttest_ind(
            depth4_win_scores,
            depth6_win_scores,
            equal_var=False
        )
        
        results['depth_ttest'] = {
            'statistic': depth_ttest.statistic,
            'p_value': depth_ttest.pvalue,
            'depth4_win_rate': np.mean(depth4_win_scores),
            'depth6_win_rate': np.mean(depth6_win_scores)
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
            
            # Compare win rates instead of game lengths
            w1_win_scores = []
            for _, game in w1_games.iterrows():
                if game['tiger_exploration_weight'] == w1:
                    score = 1 if game['tiger_won'] else 0.5 if game['draw'] else 0
                elif game['goat_exploration_weight'] == w1:
                    score = 1 if game['goat_won'] else 0.5 if game['draw'] else 0
                w1_win_scores.append(score)
            
            w2_win_scores = []
            for _, game in w2_games.iterrows():
                if game['tiger_exploration_weight'] == w2:
                    score = 1 if game['tiger_won'] else 0.5 if game['draw'] else 0
                elif game['goat_exploration_weight'] == w2:
                    score = 1 if game['goat_won'] else 0.5 if game['draw'] else 0
                w2_win_scores.append(score)
                
            weight_ttest = stats.ttest_ind(
                w1_win_scores,
                w2_win_scores,
                equal_var=False
            )
            
            weight_results[f'{w1}_vs_{w2}'] = {
                'statistic': weight_ttest.statistic,
                'p_value': weight_ttest.pvalue,
                'w1_win_rate': np.mean(w1_win_scores),
                'w2_win_rate': np.mean(w2_win_scores)
            }
    
    results['exploration_weight_ttests'] = weight_results
    
    # ANOVA for rollout policies
    policy_groups = []
    policy_win_rates = []
    policy_names = []
    
    for policy in ['random', 'lightweight', 'guided']:
        policy_games = df[(df['tiger_rollout_policy'] == policy) | (df['goat_rollout_policy'] == policy)]
        if len(policy_games) > 0:
            # Compare win rates instead of game lengths
            policy_scores = []
            for _, game in policy_games.iterrows():
                if game['tiger_rollout_policy'] == policy:
                    score = 1 if game['tiger_won'] else 0.5 if game['draw'] else 0
                elif game['goat_rollout_policy'] == policy:
                    score = 1 if game['goat_won'] else 0.5 if game['draw'] else 0
                policy_scores.append(score)
                
            policy_groups.append(policy_scores)
            policy_win_rates.append(np.mean(policy_scores))
            policy_names.append(policy)
    
    if len(policy_groups) > 1:
        policy_anova = stats.f_oneway(*policy_groups)
        
        results['policy_anova'] = {
            'statistic': policy_anova.statistic,
            'p_value': policy_anova.pvalue,
            'policies': policy_names,
            'policy_win_rates': policy_win_rates
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
    
    # Count games per configuration
    games_played = {config: 0 for config in all_configs}
    
    # Function to calculate expected score
    def expected_score(rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Function to update ratings
    def update_rating(rating, expected, actual, k_factor):
        return rating + k_factor * (actual - expected)
    
    # Function to calculate dynamic K-factor based on rating difference and games played
    def calculate_k_factor(rating_a, rating_b, games_played_a, games_played_b):
        """Calculate adaptive K-factor based on rating difference and experience"""
        # Base K-factor
        k_base = K
        
        # Reduce K slightly for configurations with more games (experience factor)
        experience_factor_a = max(0.8, 10 / (games_played_a + 5))
        experience_factor_b = max(0.8, 10 / (games_played_b + 5))
        experience_factor = (experience_factor_a + experience_factor_b) / 2
        
        # Reduce K for large rating differences (prevent volatility)
        rating_diff = abs(rating_a - rating_b)
        rating_factor = 1.0 if rating_diff < 100 else (1.0 - min(0.3, (rating_diff - 100) / 400))
        
        return k_base * experience_factor * rating_factor
    
    # Process each game
    for _, game in df.iterrows():
        tiger_config = game['tiger_config_id']
        goat_config = game['goat_config_id']
        
        # Skip if configurations are not tracked
        if tiger_config not in ratings or goat_config not in ratings:
            continue
        
        # Update games played count
        games_played[tiger_config] += 1
        games_played[goat_config] += 1
        
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
            
        # Calculate dynamic k-factor
        k_factor = calculate_k_factor(
            tiger_rating, 
            goat_rating,
            games_played[tiger_config],
            games_played[goat_config]
        )
        
        # Update ratings
        ratings[tiger_config] = update_rating(tiger_rating, tiger_expected, tiger_actual, k_factor)
        ratings[goat_config] = update_rating(goat_rating, goat_expected, goat_actual, k_factor)
    
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
                
                # Calculate confidence interval for ELO using standard error approximation
                # More games = narrower confidence interval
                confidence_range = 100 / (0.5 + 0.1 * games_played[config])
                
                elo_results.append({
                    'config_id': config,
                    'elo_rating': ratings[config],
                    'games_played': games_played[config],
                    'elo_ci_lower': ratings[config] - confidence_range,
                    'elo_ci_upper': ratings[config] + confidence_range,
                    'rollout_policy': rollout_policy,
                    'rollout_depth': rollout_depth,
                    'exploration_weight': exploration_weight
                })
    
    elo_df = pd.DataFrame(elo_results)
    elo_df = elo_df.sort_values('elo_rating', ascending=False)
    
    return elo_df

def generate_composite_scores(win_rates_df, elo_df, win_rate_weight=0.5, elo_weight=0.5):
    """
    Generate composite scores from win rates and Elo ratings.
    
    Args:
        win_rates_df: DataFrame with win rates
        elo_df: DataFrame with Elo ratings
        win_rate_weight: Weight for win rate in composite score (default 0.5)
        elo_weight: Weight for Elo rating in composite score (default 0.5)
        
    Returns:
        DataFrame with composite scores
    """
    # Merge win rates and Elo ratings
    merged_df = pd.merge(
        win_rates_df,
        elo_df[['config_id', 'elo_rating', 'elo_ci_lower', 'elo_ci_upper']],
        on='config_id',
        how='inner'
    )
    
    # Normalize Elo ratings to 0-1 scale
    min_elo = merged_df['elo_rating'].min()
    max_elo = merged_df['elo_rating'].max()
    merged_df['normalized_elo'] = (merged_df['elo_rating'] - min_elo) / (max_elo - min_elo) if max_elo > min_elo else 0.5
    
    # Calculate composite score (weighted average of adjusted win rate and normalized Elo)
    merged_df['composite_score'] = (
        win_rate_weight * merged_df['adjusted_win_rate'] + 
        elo_weight * merged_df['normalized_elo']
    )
    
    # Sort by composite score
    merged_df = merged_df.sort_values('composite_score', ascending=False)
    
    return merged_df

def select_top_configurations(composite_scores_df, n=3, enforce_diversity=True):
    """
    Select top n configurations considering diversity.
    
    Args:
        composite_scores_df: DataFrame with composite scores
        n: Number of configurations to select
        enforce_diversity: Whether to promote diversity in selection
        
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
        
        # Add confidence intervals if available
        if 'adjusted_win_rate_ci_lower' in top_candidate:
            config['win_rate_ci_lower'] = float(top_candidate['adjusted_win_rate_ci_lower'])
            config['win_rate_ci_upper'] = float(top_candidate['adjusted_win_rate_ci_upper'])
            
        if 'elo_ci_lower' in top_candidate:
            config['elo_ci_lower'] = float(top_candidate['elo_ci_lower'])
            config['elo_ci_upper'] = float(top_candidate['elo_ci_upper'])
        
        # If we already have n-1 configurations, just add the best remaining one
        if len(top_configs) == n - 1:
            top_configs.append(config)
            break
            
        # Otherwise, check for diversity if required
        if not enforce_diversity or (top_candidate['rollout_policy'] not in selected_policies or 
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
        
        # Add confidence intervals if available
        if 'adjusted_win_rate_ci_lower' in top_candidate:
            config['win_rate_ci_lower'] = float(top_candidate['adjusted_win_rate_ci_lower'])
            config['win_rate_ci_upper'] = float(top_candidate['adjusted_win_rate_ci_upper'])
            
        if 'elo_ci_lower' in top_candidate:
            config['elo_ci_lower'] = float(top_candidate['elo_ci_lower'])
            config['elo_ci_upper'] = float(top_candidate['elo_ci_upper'])
        
        top_configs.append(config)
        candidates = candidates.iloc[1:]
    
    return top_configs 