"""
Visualization functions for MCTS tournament analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import to_rgba

def ensure_directory(path):
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)

def create_win_rate_bar_chart(win_rates_df, output_dir, config=None):
    """
    Create bar chart of win rates for top 10 configurations.
    
    Args:
        win_rates_df: DataFrame with win rates
        output_dir: Directory to save output figure
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Get top 10 configurations by adjusted win rate
    top_10 = win_rates_df.head(10).copy()
    
    # Create a color mapping for policies
    policy_colors = {
        'random': 'skyblue',
        'lightweight': 'lightgreen',
        'guided': 'coral'
    }
    
    # Create a simplified configuration label
    top_10['config_label'] = top_10.apply(
        lambda row: f"{row['rollout_policy']}\nd={row['rollout_depth']}\ne={row['exploration_weight']}",
        axis=1
    )
    
    # Create the bar colors based on policy
    bar_colors = [policy_colors.get(policy, 'gray') for policy in top_10['rollout_policy']]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create numeric positions for the bars
    x_positions = np.arange(len(top_10))
    
    # Plot overall adjusted win rate (includes draws as 0.5 points)
    bars = plt.bar(
        x_positions,
        top_10['adjusted_win_rate'],
        width=0.7,  # Reduced width for better spacing
        color=bar_colors,
        alpha=0.7
    )
    
    # Add confidence intervals if available
    if 'adjusted_win_rate_ci_lower' in top_10.columns and 'adjusted_win_rate_ci_upper' in top_10.columns:
        plt.errorbar(
            x_positions,
            top_10['adjusted_win_rate'],
            yerr=[
                top_10['adjusted_win_rate'] - top_10['adjusted_win_rate_ci_lower'],
                top_10['adjusted_win_rate_ci_upper'] - top_10['adjusted_win_rate']
            ],
            fmt='none',
            ecolor='black',
            capsize=5,
            alpha=0.7
        )
    
    # Add win rate as tiger and goat as error bars
    for i, (_, row) in enumerate(top_10.iterrows()):
        plt.plot(
            [i, i],
            [row['adjusted_win_rate_as_tiger'], row['adjusted_win_rate_as_goat']],
            color='black',
            linestyle='-',
            linewidth=2,
            alpha=0.6
        )
        
        # Add markers for tiger and goat win rates
        plt.plot(i, row['adjusted_win_rate_as_tiger'], 'ro', markersize=5, label='Tiger' if i == 0 else "")
        plt.plot(i, row['adjusted_win_rate_as_goat'], 'bo', markersize=5, label='Goat' if i == 0 else "")
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Configuration')
    plt.ylabel('Win Rate (draws = 0.5 points)')
    plt.title('Top 10 MCTS Configurations by Win Rate (Time-Constrained: 20s per move, draws = 0.5 points)')
    plt.ylim(0, 1)
    
    # Add legend for the first iteration only to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Add policy legend
    policy_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7)
        for policy, color in policy_colors.items()
    ]
    plt.legend(
        policy_patches + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8),
                         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8)],
        list(policy_colors.keys()) + ['Tiger', 'Goat'],
        loc='upper right'
    )
    
    # Important: This makes the x-tick label align with the center of the bar
    plt.xticks(
        x_positions,  # Use the exact same positions as the bars
        top_10['config_label'], 
        rotation=45, 
        ha='right',  # Changed to 'right' for proper anchor point with anchor rotation
        va='top',    # Added to align top of text with tick
        fontsize=7  # Smaller font
    )
    
    # Set rotation mode to anchor at the end of the text for better alignment
    plt.setp(plt.gca().get_xticklabels(), rotation_mode="anchor")
    plt.gca().set_xticks(x_positions, minor=False)
    
    # Add data labels - positioned to the left of each bar to avoid intersecting with vertical variance lines
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()*0.25,  # Position at 1/4 of the bar width (more to the left)
            height + 0.02,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            rotation=0,
            fontsize=8,
            color='black'
        )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # More space at the bottom for x labels
    
    # Save the figure
    output_path = os.path.join(output_dir, 'win_rates.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Win rate chart saved to {output_path}")

def create_parameter_performance_charts(df, stats_results, output_dir, config=None):
    """
    Create bar charts for each parameter's performance.
    
    Args:
        df: Preprocessed tournament data
        stats_results: Dictionary with statistical test results
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Set significance threshold
    significance_threshold = 0.05
    if config and 'statistical' in config and 'significance_threshold' in config['statistical']:
        significance_threshold = config['statistical']['significance_threshold']
    
    # 1. Rollout Depth Chart
    plt.figure(figsize=(8, 6))
    
    # Always extract unique depth values from data
    depths = sorted(set(df['tiger_rollout_depth'].unique()) | set(df['goat_rollout_depth'].unique()))
    
    # Calculate win rates by depth
    depth_win_rates = []
    
    for depth in depths:
        # Games where either tiger or goat used this depth
        depth_games = df[(df['tiger_rollout_depth'] == depth) | (df['goat_rollout_depth'] == depth)]
        
        # Skip if no games with this depth
        if len(depth_games) == 0:
            continue
        
        # Win rate when playing as tiger with this depth
        tiger_games = df[df['tiger_rollout_depth'] == depth]
        tiger_win_rate = tiger_games['tiger_won'].mean() if len(tiger_games) > 0 else 0
        tiger_draw_rate = tiger_games['draw'].mean() if len(tiger_games) > 0 else 0
        tiger_adjusted_win_rate = tiger_win_rate + 0.5 * tiger_draw_rate  # Count draws as 0.5 points
        
        # Win rate when playing as goat with this depth
        goat_games = df[df['goat_rollout_depth'] == depth]
        goat_win_rate = goat_games['goat_won'].mean() if len(goat_games) > 0 else 0
        goat_draw_rate = goat_games['draw'].mean() if len(goat_games) > 0 else 0
        goat_adjusted_win_rate = goat_win_rate + 0.5 * goat_draw_rate  # Count draws as 0.5 points
        
        # Overall win rate
        total_games = len(tiger_games) + len(goat_games)
        if total_games > 0:
            total_wins = tiger_games['tiger_won'].sum() + goat_games['goat_won'].sum()
            total_draws = tiger_games['draw'].sum() + goat_games['draw'].sum()
            overall_adjusted_win_rate = (total_wins + 0.5 * total_draws) / total_games
        else:
            overall_adjusted_win_rate = 0
        
        # Calculate confidence intervals
        tiger_ci = (0, 0)
        goat_ci = (0, 0)
        overall_ci = (0, 0)
        
        if len(tiger_games) > 0:
            from .mcts_analysis import calculate_confidence_intervals
            tiger_ci = calculate_confidence_intervals(tiger_adjusted_win_rate, len(tiger_games))
        
        if len(goat_games) > 0:
            from .mcts_analysis import calculate_confidence_intervals
            goat_ci = calculate_confidence_intervals(goat_adjusted_win_rate, len(goat_games))
            
        if total_games > 0:
            from .mcts_analysis import calculate_confidence_intervals
            overall_ci = calculate_confidence_intervals(overall_adjusted_win_rate, total_games)
        
        depth_win_rates.append({
            'depth': depth,
            'tiger_adjusted_win_rate': tiger_adjusted_win_rate,
            'goat_adjusted_win_rate': goat_adjusted_win_rate,
            'overall_adjusted_win_rate': overall_adjusted_win_rate,
            'game_count': total_games,
            'tiger_ci_lower': tiger_ci[0],
            'tiger_ci_upper': tiger_ci[1],
            'goat_ci_lower': goat_ci[0],
            'goat_ci_upper': goat_ci[1],
            'overall_ci_lower': overall_ci[0],
            'overall_ci_upper': overall_ci[1]
        })
    
    depth_df = pd.DataFrame(depth_win_rates)
    
    # If no data, skip this chart
    if len(depth_df) == 0:
        print("No depth data available for chart")
        return
    
    # Width of the bars
    width = 0.3
    
    # Set position of bar on X axis
    r1 = np.arange(len(depth_df))
    r2 = [x + width for x in r1]
    
    # Make the plot for depth chart
    tiger_bars = plt.bar(r1, depth_df['tiger_adjusted_win_rate'], width=width, color='indianred', label='As Tiger')
    goat_bars = plt.bar(r2, depth_df['goat_adjusted_win_rate'], width=width, color='royalblue', label='As Goat')
    
    # Add confidence intervals
    plt.errorbar(
        r1, 
        depth_df['tiger_adjusted_win_rate'],
        yerr=[
            depth_df['tiger_adjusted_win_rate'] - depth_df['tiger_ci_lower'],
            depth_df['tiger_ci_upper'] - depth_df['tiger_adjusted_win_rate']
        ],
        fmt='none',
        ecolor='black',
        capsize=4,
        alpha=0.7
    )
    
    plt.errorbar(
        r2, 
        depth_df['goat_adjusted_win_rate'],
        yerr=[
            depth_df['goat_adjusted_win_rate'] - depth_df['goat_ci_lower'],
            depth_df['goat_ci_upper'] - depth_df['goat_adjusted_win_rate']
        ],
        fmt='none',
        ecolor='black',
        capsize=4,
        alpha=0.7
    )
    
    # Add value labels to the top of each bar - positioned more to the left to avoid overlap with error bars
    for i, bar in enumerate(tiger_bars):
        value = depth_df['tiger_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.25,  # Position at 25% of the bar width (slightly less to the left)
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    for i, bar in enumerate(goat_bars):
        value = depth_df['goat_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.25,  # Position at 25% of the bar width (slightly less to the left)
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    # Add labels and title
    plt.xlabel('Rollout Depth')
    plt.ylabel('Win Rate (draws = 0.5 points)')
    plt.title('Win Rate by Rollout Depth (Time-Constrained: 20s per move, draws = 0.5 points)')
    plt.xticks([r + width/2 for r in range(len(depth_df))], depth_df['depth'])
    plt.ylim(0, 1)
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add text with t-test results if available
    if 'depth_ttest' in stats_results:
        p_value = stats_results['depth_ttest']['p_value']
        is_significant = p_value < significance_threshold
        significance_text = "★ Statistically significant difference" if is_significant else "No statistically significant difference"
        
        plt.text(
            0.5, 0.9,
            f"T-test p-value: {p_value:.4f}\n{significance_text}",
            transform=plt.gca().transAxes,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'depth_performance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Depth performance chart saved to {output_path}")
    
    # 2. Exploration Weight Chart
    plt.figure(figsize=(10, 6))
    
    # Extract unique exploration weights
    weights = sorted(set(df['tiger_exploration_weight'].unique()) | set(df['goat_exploration_weight'].unique()))
    
    # Calculate win rates by exploration weight
    weight_win_rates = []
    
    for weight in weights:
        # Games where either tiger or goat used this weight
        weight_games = df[(df['tiger_exploration_weight'] == weight) | (df['goat_exploration_weight'] == weight)]
        
        # Win rate when playing as tiger with this weight
        tiger_games = df[df['tiger_exploration_weight'] == weight]
        tiger_win_rate = tiger_games['tiger_won'].mean() if len(tiger_games) > 0 else 0
        tiger_draw_rate = tiger_games['draw'].mean() if len(tiger_games) > 0 else 0
        tiger_adjusted_win_rate = tiger_win_rate + 0.5 * tiger_draw_rate  # Count draws as 0.5 points
        
        # Win rate when playing as goat with this weight
        goat_games = df[df['goat_exploration_weight'] == weight]
        goat_win_rate = goat_games['goat_won'].mean() if len(goat_games) > 0 else 0
        goat_draw_rate = goat_games['draw'].mean() if len(goat_games) > 0 else 0
        goat_adjusted_win_rate = goat_win_rate + 0.5 * goat_draw_rate  # Count draws as 0.5 points
        
        # Overall win rate
        total_games = len(tiger_games) + len(goat_games)
        if total_games > 0:
            total_wins = tiger_games['tiger_won'].sum() + goat_games['goat_won'].sum()
            total_draws = tiger_games['draw'].sum() + goat_games['draw'].sum()
            overall_adjusted_win_rate = (total_wins + 0.5 * total_draws) / total_games
        else:
            overall_adjusted_win_rate = 0
        
        weight_win_rates.append({
            'weight': weight,
            'tiger_adjusted_win_rate': tiger_adjusted_win_rate,
            'goat_adjusted_win_rate': goat_adjusted_win_rate,
            'overall_adjusted_win_rate': overall_adjusted_win_rate,
            'game_count': total_games
        })
    
    weight_df = pd.DataFrame(weight_win_rates)
    
    # Width of the bars
    width = 0.3
    
    # Set position of bar on X axis
    r1 = np.arange(len(weights))
    r2 = [x + width for x in r1]
    
    # Make the plot for exploration weight chart
    tiger_bars = plt.bar(r1, weight_df['tiger_adjusted_win_rate'], width=width, color='indianred', label='As Tiger')
    goat_bars = plt.bar(r2, weight_df['goat_adjusted_win_rate'], width=width, color='royalblue', label='As Goat')
    
    # Add value labels to the top of each bar - positioned at center
    for i, bar in enumerate(tiger_bars):
        value = weight_df['tiger_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.5,  # Position at center of the bar
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    for i, bar in enumerate(goat_bars):
        value = weight_df['goat_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.5,  # Position at center of the bar
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    # Add labels and title
    plt.xlabel('Exploration Weight')
    plt.ylabel('Win Rate (draws = 0.5 points)')
    plt.title('Win Rate by Exploration Weight (Time-Constrained: 20s per move, draws = 0.5 points)')
    plt.xticks([r + width/2 for r in range(len(weights))], weights)
    plt.ylim(0, 1)
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add text with t-test results if available
    if 'exploration_weight_ttests' in stats_results and stats_results['exploration_weight_ttests']:
        text_lines = []
        for key, result in stats_results['exploration_weight_ttests'].items():
            p_value = result['p_value']
            text_lines.append(
                f"{key}: p={p_value:.4f}" +
                (" *" if p_value < 0.05 else "")
            )
        
        plt.text(
            0.02, 0.85,  # Position lower at 85% instead of 95% vertical position
            "\n".join(text_lines),
            transform=plt.gca().transAxes,
            ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)  # Reduced padding to match legend style
        )
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'exploration_performance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Exploration weight performance chart saved to {output_path}")
    
    # 3. Rollout Policy Chart
    plt.figure(figsize=(10, 6))
    
    # Extract unique policies
    policies = sorted(set(df['tiger_rollout_policy'].unique()) | set(df['goat_rollout_policy'].unique()))
    
    # Calculate win rates by policy
    policy_win_rates = []
    
    for policy in policies:
        # Games where either tiger or goat used this policy
        policy_games = df[(df['tiger_rollout_policy'] == policy) | (df['goat_rollout_policy'] == policy)]
        
        # Win rate when playing as tiger with this policy
        tiger_games = df[df['tiger_rollout_policy'] == policy]
        tiger_win_rate = tiger_games['tiger_won'].mean() if len(tiger_games) > 0 else 0
        tiger_draw_rate = tiger_games['draw'].mean() if len(tiger_games) > 0 else 0
        tiger_adjusted_win_rate = tiger_win_rate + 0.5 * tiger_draw_rate  # Count draws as 0.5 points
        
        # Win rate when playing as goat with this policy
        goat_games = df[df['goat_rollout_policy'] == policy]
        goat_win_rate = goat_games['goat_won'].mean() if len(goat_games) > 0 else 0
        goat_draw_rate = goat_games['draw'].mean() if len(goat_games) > 0 else 0
        goat_adjusted_win_rate = goat_win_rate + 0.5 * goat_draw_rate  # Count draws as 0.5 points
        
        # Overall win rate
        total_games = len(tiger_games) + len(goat_games)
        if total_games > 0:
            total_wins = tiger_games['tiger_won'].sum() + goat_games['goat_won'].sum()
            total_draws = tiger_games['draw'].sum() + goat_games['draw'].sum()
            overall_adjusted_win_rate = (total_wins + 0.5 * total_draws) / total_games
        else:
            overall_adjusted_win_rate = 0
        
        policy_win_rates.append({
            'policy': policy,
            'tiger_adjusted_win_rate': tiger_adjusted_win_rate,
            'goat_adjusted_win_rate': goat_adjusted_win_rate,
            'overall_adjusted_win_rate': overall_adjusted_win_rate,
            'game_count': total_games
        })
    
    policy_df = pd.DataFrame(policy_win_rates)
    
    # Width of the bars
    width = 0.3
    
    # Set position of bar on X axis
    r1 = np.arange(len(policies))
    r2 = [x + width for x in r1]
    
    # Make the plot for policy chart
    tiger_bars = plt.bar(r1, policy_df['tiger_adjusted_win_rate'], width=width, color='indianred', label='As Tiger')
    goat_bars = plt.bar(r2, policy_df['goat_adjusted_win_rate'], width=width, color='royalblue', label='As Goat')
    
    # Add value labels to the top of each bar - positioned at center
    for i, bar in enumerate(tiger_bars):
        value = policy_df['tiger_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.5,  # Position at center of the bar
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    for i, bar in enumerate(goat_bars):
        value = policy_df['goat_adjusted_win_rate'].iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()*0.5,  # Position at center of the bar
            value + 0.02, 
            f'{value:.2f}', 
            ha='center', 
            va='bottom', 
            fontsize=8
        )
    
    # Add labels and title
    plt.xlabel('Rollout Policy')
    plt.ylabel('Win Rate (draws = 0.5 points)')
    plt.title('Win Rate by Rollout Policy (Time-Constrained: 20s per move, draws = 0.5 points)')
    plt.xticks([r + width/2 for r in range(len(policies))], policies)
    plt.ylim(0, 1)
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add text with ANOVA results if available
    if 'policy_anova' in stats_results:
        p_value = stats_results['policy_anova']['p_value']
        plt.text(
            0.5, 0.9,
            f"ANOVA p-value: {p_value:.4f}" +
            (" (significant)" if p_value < 0.05 else ""),
            transform=plt.gca().transAxes,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'policy_performance.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Policy performance chart saved to {output_path}")

def create_elo_rating_chart(elo_df, output_dir, config=None):
    """
    Create bar chart of Elo ratings for all configurations.
    
    Args:
        elo_df: DataFrame with Elo ratings
        output_dir: Directory to save output figure
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Create a color mapping for policies
    policy_colors = {
        'random': 'skyblue',
        'lightweight': 'lightgreen',
        'guided': 'coral',
    }
    
    # Create a simplified configuration label
    elo_df['config_label'] = elo_df.apply(
        lambda row: f"{row['rollout_policy']}\nd={row['rollout_depth']}\ne={row['exploration_weight']}" + 
                   (f"\ns={row['guided_strictness']}" if row['rollout_policy'] == 'guided' and 'guided_strictness' in row and pd.notna(row['guided_strictness']) else ""),
        axis=1
    )
    
    # Create the bar colors based on policy
    bar_colors = [policy_colors.get(policy, 'gray') for policy in elo_df['rollout_policy']]
    
    # Create the plot with wider spacing to avoid overlapping labels
    plt.figure(figsize=(14, 6))  # Increased width for better spacing
    
    # Create numeric positions for the bars
    x_positions = np.arange(len(elo_df))
    
    # Plot Elo ratings with numeric x positions for better control
    width = 0.7  # Reduced bar width to create more space between them
    bars = plt.bar(
        x_positions,  # Use numeric indices for x-axis positioning
        elo_df['elo_rating'],
        width=width,
        color=bar_colors,
        alpha=0.7
    )
    
    # Note: Confidence intervals removed as requested
    
    # Add a horizontal line for initial rating
    initial_rating = 1500
    if config and 'elo' in config and 'initial_rating' in config['elo']:
        initial_rating = config['elo']['initial_rating']
    plt.axhline(y=initial_rating, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Configuration')
    plt.ylabel('Elo Rating')
    plt.title('MCTS Configurations by Elo Rating (Time-Constrained: 20s per move)')
    
    # Add policy legend
    policy_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7)
        for policy, color in policy_colors.items()
    ]
    plt.legend(
        policy_patches,
        list(policy_colors.keys()),
        loc='upper right'
    )
    
    # Important: This makes the x-tick label align with the center of the bar
    plt.xticks(
        x_positions,  # Use the exact same positions as the bars
        elo_df['config_label'], 
        rotation=45, 
        ha='right',  # Changed to 'right' for proper anchor point with anchor rotation
        va='top',    # Added to align top of text with tick
        fontsize=7  # Smaller font
    )
    
    # Set rotation mode to anchor at the end of the text for better alignment
    plt.setp(plt.gca().get_xticklabels(), rotation_mode="anchor")
    plt.gca().set_xticks(x_positions, minor=False)
    
    # Add data labels - positioned at the center of the bar since there are no confidence intervals
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()*0.5,  # Position at center of bar width
            height + 5,  # Position above the bar
            f'{height:.0f}',
            ha='center',
            va='bottom',
            rotation=0,
            fontsize=8,
            color='black'
        )
    
    # Adjust tight_layout to leave more space for labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # More space at the bottom for x labels
    
    # Save the figure
    output_path = os.path.join(output_dir, 'elo_ratings.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Elo rating chart saved to {output_path}")

def create_composite_score_chart(composite_df, top_configs, output_dir, config=None):
    """
    Create bar chart of composite scores for all configurations.
    
    Args:
        composite_df: DataFrame with composite scores
        top_configs: List of top configurations
        output_dir: Directory to save output figure
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Create a color mapping for policies
    policy_colors = {
        'random': 'skyblue',
        'lightweight': 'lightgreen',
        'guided': 'coral'
    }
    
    # Create a simplified configuration label
    composite_df['config_label'] = composite_df.apply(
        lambda row: f"{row['rollout_policy']}\nd={row['rollout_depth']}\ne={row['exploration_weight']}" + 
                   (f"\ns={row['guided_strictness']}" if row['rollout_policy'] == 'guided' and 'guided_strictness' in row and pd.notna(row['guided_strictness']) else ""),
        axis=1
    )
    
    # Mark top configurations
    top_config_ids = [config['config_id'] for config in top_configs]
    composite_df['is_top'] = composite_df['config_id'].isin(top_config_ids)
    
    # Sort by composite score
    composite_df = composite_df.sort_values('composite_score', ascending=False)
    
    # Create the bar colors based on policy and highlight top configs
    bar_colors = []
    for i, row in composite_df.iterrows():
        base_color = policy_colors.get(row['rollout_policy'], 'gray')
        # Make top configurations more saturated
        if row['is_top']:
            bar_colors.append(base_color)
        else:
            # Create a lighter version for non-top configs
            bar_colors.append(to_rgba(base_color, alpha=0.5))
    
    # Create the plot with wider spacing to avoid overlapping labels
    plt.figure(figsize=(14, 6))  # Increased width for better spacing
    
    # Plot composite scores with wider bars and spacing
    width = 0.7  # Reduced bar width to create more space between them
    
    # Create numeric positions for the bars
    x_positions = np.arange(len(composite_df))
    
    bars = plt.bar(
        x_positions,  # Use numeric indices for x-axis positioning
        composite_df['composite_score'],
        width=width,
        color=bar_colors
    )
    
    # Add confidence intervals if available
    has_confidence = ('adjusted_win_rate_ci_lower' in composite_df.columns and 
                     'adjusted_win_rate_ci_upper' in composite_df.columns and
                     'elo_ci_lower' in composite_df.columns and
                     'elo_ci_upper' in composite_df.columns)
                     
    if has_confidence:
        # Get weights for composite score components
        win_rate_weight = 0.5
        elo_weight = 0.5
        if config and 'composite_score' in config:
            win_rate_weight = config['composite_score'].get('win_rate_weight', 0.5)
            elo_weight = config['composite_score'].get('elo_weight', 0.5)
            
        # Normalize ELO CI bounds
        min_elo = composite_df['elo_rating'].min()
        max_elo = composite_df['elo_rating'].max()
        elo_range = max_elo - min_elo if max_elo > min_elo else 1
        
        # Calculate lower and upper bounds for composite score
        composite_df['composite_ci_lower'] = (
            win_rate_weight * composite_df['adjusted_win_rate_ci_lower'] + 
            elo_weight * ((composite_df['elo_ci_lower'] - min_elo) / elo_range)
        )
        
        composite_df['composite_ci_upper'] = (
            win_rate_weight * composite_df['adjusted_win_rate_ci_upper'] + 
            elo_weight * ((composite_df['elo_ci_upper'] - min_elo) / elo_range)
        )
        
        # Add error bars
        plt.errorbar(
            x_positions,
            composite_df['composite_score'],
            yerr=[
                composite_df['composite_score'] - composite_df['composite_ci_lower'],
                composite_df['composite_ci_upper'] - composite_df['composite_score']
            ],
            fmt='none',
            ecolor='black',
            capsize=4,
            alpha=0.7
        )
    
    # Add labels and title
    plt.xlabel('Configuration')
    plt.ylabel('Composite Score')
    title = 'MCTS Configurations by Composite Score (Time-Constrained: 20s per move)'
    
    # Add weights to title if available from config
    if config and 'composite_score' in config:
        w1 = config['composite_score'].get('win_rate_weight', 0.5)
        w2 = config['composite_score'].get('elo_weight', 0.5)
        title += f'\n(Win Rate Weight: {w1}, Elo Weight: {w2})'
        
    plt.title(title)
    
    # Important: This makes the x-tick label align with the center of the bar
    plt.xticks(
        x_positions,  # Use the exact same positions as the bars
        composite_df['config_label'], 
        rotation=45, 
        ha='right',  # Changed to 'right' for proper anchor point with anchor rotation
        va='top',    # Added to align top of text with tick
        fontsize=7  # Smaller font
    )
    
    # Set rotation mode to anchor at the end of the text for better alignment
    plt.setp(plt.gca().get_xticklabels(), rotation_mode="anchor")
    plt.gca().set_xticks(x_positions, minor=False)
    
    # Mark top configurations with a star or highlight
    for i, is_top in enumerate(composite_df['is_top']):
        if is_top:
            plt.plot(i, composite_df.iloc[i]['composite_score'] + 0.02, 'k*', markersize=10)
    
    # Add data labels - positioned even more to the left for better visibility with confidence intervals
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()*0.1,  # Position at 10% of the bar width (even more to the left)
            height + 0.02,  # Position above the bar
            f'{height:.2f}',
            ha='center',
            va='bottom',
            rotation=0,
            fontsize=8,
            color='black'
        )
    
    # Add a legend for top configurations
    top_patch = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
                          markersize=10, label='Top Configurations')
    
    # Add policy legend
    policy_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for policy, color in policy_colors.items()
    ]
    
    handles = policy_patches + [top_patch]
    labels = list(policy_colors.keys()) + ['Top Configurations']
    
    plt.legend(handles, labels, loc='upper right')
    
    # Adjust tight_layout to leave more space for labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # More space at the bottom for x labels
    
    # Save the figure
    output_path = os.path.join(output_dir, 'composite_scores.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Composite score chart saved to {output_path}")

def create_heatmap(df, output_dir):
    """
    Create heat map of parameter interactions.
    
    Args:
        df: Preprocessed tournament data
        output_dir: Directory to save output figure
    """
    ensure_directory(output_dir)
    
    # Get unique rollout policies
    policies = sorted(set(df['tiger_rollout_policy'].unique()) | set(df['goat_rollout_policy'].unique()))
    
    # Get unique rollout depths
    depths = sorted(set(df['tiger_rollout_depth'].unique()) | set(df['goat_rollout_depth'].unique()))
    
    # Get unique exploration weights
    weights = sorted(set(df['tiger_exploration_weight'].unique()) | set(df['goat_exploration_weight'].unique()))
    
    # Create the figure with extra bottom margin to prevent clipping
    fig, axes = plt.subplots(1, len(policies), figsize=(15, 5.5), sharey=True)
    if len(policies) == 1:
        axes = [axes]
    
    # For each policy, create a heatmap
    for i, policy in enumerate(policies):
        # Initialize a matrix for the heatmap
        heatmap_data = np.zeros((len(depths), len(weights)))
        game_counts = np.zeros((len(depths), len(weights)))
        
        # Extract games for this policy
        policy_games_tiger = df[df['tiger_rollout_policy'] == policy]
        policy_games_goat = df[df['goat_rollout_policy'] == policy]
        
        # Calculate win rates for each depth-weight combination
        for d_idx, depth in enumerate(depths):
            for w_idx, weight in enumerate(weights):
                # Tiger perspective
                tiger_games = policy_games_tiger[
                    (policy_games_tiger['tiger_rollout_depth'] == depth) &
                    (policy_games_tiger['tiger_exploration_weight'] == weight)
                ]
                tiger_wins = tiger_games['tiger_won'].sum() if len(tiger_games) > 0 else 0
                tiger_draws = tiger_games['draw'].sum() if len(tiger_games) > 0 else 0
                
                # Goat perspective
                goat_games = policy_games_goat[
                    (policy_games_goat['goat_rollout_depth'] == depth) &
                    (policy_games_goat['goat_exploration_weight'] == weight)
                ]
                goat_wins = goat_games['goat_won'].sum() if len(goat_games) > 0 else 0
                goat_draws = goat_games['draw'].sum() if len(goat_games) > 0 else 0
                
                # Total games for this configuration
                total_games = len(tiger_games) + len(goat_games)
                if total_games > 0:
                    # Count draws as 0.5 points for adjusted win rate
                    win_rate = (tiger_wins + goat_wins + 0.5 * (tiger_draws + goat_draws)) / total_games
                    heatmap_data[d_idx, w_idx] = win_rate
                    game_counts[d_idx, w_idx] = total_games
        
        # Create the heatmap
        im = axes[i].imshow(
            heatmap_data,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            aspect='auto'
        )
        
        # Add labels
        axes[i].set_xticks(np.arange(len(weights)))
        axes[i].set_yticks(np.arange(len(depths)))
        axes[i].set_xticklabels(weights)
        axes[i].set_yticklabels(depths)
        
        # Add title
        axes[i].set_title(f"Policy: {policy}")
        
        # Set axis labels (only for the first subplot for y-axis)
        if i == 0:
            axes[i].set_ylabel("Rollout Depth")
        axes[i].set_xlabel("Exploration Weight")
        
        # Add text annotations with win rates and game counts
        for d_idx in range(len(depths)):
            for w_idx in range(len(weights)):
                if game_counts[d_idx, w_idx] > 0:
                    win_rate = heatmap_data[d_idx, w_idx]
                    axes[i].text(
                        w_idx, d_idx,
                        f"{win_rate:.2f}\n({int(game_counts[d_idx, w_idx])})",
                        ha="center", va="center",
                        color="black" if win_rate < 0.7 else "white",
                        fontsize=8
                    )
    
    # Add colorbar with reduced height to avoid overlapping with the heatmap
    cbar_ax = fig.add_axes([0.3, 0.08, 0.4, 0.03])  # Adjusted position to avoid overlap
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Win Rate (Time-Constrained: 20s per move, draws = 0.5 points)')
    
    plt.suptitle('MCTS Configuration Performance Heatmap (draws = 0.5 points)')
    plt.tight_layout(rect=[0, 0.15, 1, 0.97])  # Adjust the figure layout to make room for the colorbar
    
    # Save the figure
    output_path = os.path.join(output_dir, 'parameter_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Parameter heatmap saved to {output_path}") 