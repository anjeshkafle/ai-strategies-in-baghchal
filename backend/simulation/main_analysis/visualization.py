"""
Visualization functions for MCTS vs Minimax competition analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec

def ensure_directory(path):
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)

def create_win_rate_visualizations(performance_metrics, output_dir, config=None):
    """
    Create visualizations of algorithm win rates.
    
    Args:
        performance_metrics: Dictionary with performance metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    algorithm_comparison = performance_metrics['algorithm_comparison']
    role_performance = performance_metrics['role_performance']
    
    # Set colors from config or use defaults
    color_mcts = config['visualization']['color_mcts'] if config else "#1f77b4"
    color_minimax = config['visualization']['color_minimax'] if config else "#ff7f0e"
    color_tiger = config['visualization']['color_tiger'] if config else "#d62728"
    color_goat = config['visualization']['color_goat'] if config else "#2ca02c"
    
    # Plot 1: Overall Algorithm Win Rates with Confidence Intervals
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    positions = np.arange(len(algorithm_comparison))
    
    # Plot bars
    bars = plt.bar(
        positions,
        algorithm_comparison['Win Rate'],
        yerr=[
            algorithm_comparison['Win Rate'] - algorithm_comparison['CI Lower'],
            algorithm_comparison['CI Upper'] - algorithm_comparison['Win Rate']
        ],
        width=0.6,
        capsize=10,
        color=[color_mcts, color_minimax],
        alpha=0.7
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
    plt.title('Overall Win Rates by Algorithm', fontsize=16)
    plt.ylim(0, 1)
    
    # Set x-ticks
    plt.xticks(positions, algorithm_comparison['Algorithm'], fontsize=12)
    
    # Add number of games as text
    for i, games in enumerate(algorithm_comparison['Games']):
        plt.text(
            positions[i],
            0.05,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_win_rates.png'), dpi=300)
    plt.close()
    
    # Plot 2: Win Rates by Role
    plt.figure(figsize=(12, 7))
    
    # Create bar positions
    positions = np.arange(len(role_performance))
    bar_width = 0.25
    
    # Create bars for each outcome
    plt.bar(
        positions - bar_width,
        role_performance['Win %'],
        width=bar_width,
        color='green',
        alpha=0.7,
        label='Win'
    )
    
    plt.bar(
        positions,
        role_performance['Draw %'],
        width=bar_width,
        color='gray',
        alpha=0.7,
        label='Draw'
    )
    
    plt.bar(
        positions + bar_width,
        role_performance['Loss %'],
        width=bar_width,
        color='red',
        alpha=0.7,
        label='Loss'
    )
    
    # Add error bars for adjusted win rate
    plt.errorbar(
        positions,
        role_performance['Win Rate'],
        yerr=[
            role_performance['Win Rate'] - role_performance['CI Lower'],
            role_performance['CI Upper'] - role_performance['Win Rate']
        ],
        fmt='o',
        color='black',
        capsize=5,
        label='Adjusted Win Rate (with 95% CI)'
    )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Set labels and title
    plt.xlabel('Role', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.title('Performance by Algorithm and Role', fontsize=16)
    plt.ylim(0, 1)
    
    # Set x-ticks
    plt.xticks(positions, role_performance['Role'], fontsize=12)
    
    # Add number of games as text
    for i, games in enumerate(role_performance['Games']):
        plt.text(
            positions[i],
            0.05,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'role_performance.png'), dpi=300)
    plt.close()
    
    # Plot 3: Tiger vs Goat Performance for each Algorithm
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create data for grouped bar chart
    algorithms = ['MCTS', 'Minimax']
    tiger_rates = [performance_metrics['mcts_as_tiger_win_rate'], performance_metrics['minimax_as_tiger_win_rate']]
    goat_rates = [performance_metrics['mcts_as_goat_win_rate'], performance_metrics['minimax_as_goat_win_rate']]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    # Create bars
    tiger_bars = ax.bar(x - width/2, tiger_rates, width, label='As Tiger', color=color_tiger, alpha=0.7)
    goat_bars = ax.bar(x + width/2, goat_rates, width, label='As Goat', color=color_goat, alpha=0.7)
    
    # Add a horizontal line for 50% win rate
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add data labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
    add_labels(tiger_bars)
    add_labels(goat_bars)
    
    # Set labels and title
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel('Win Rate (draws = 0.5)', fontsize=14)
    ax.set_title('Win Rates by Algorithm and Role', fontsize=16)
    ax.set_ylim(0, 1)
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_role_win_rates.png'), dpi=300)
    plt.close()

def create_depth_performance_visualizations(performance_metrics, output_dir, config=None):
    """
    Create visualizations of Minimax depth performance.
    
    Args:
        performance_metrics: Dictionary with performance metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    depth_performance = performance_metrics['depth_performance']
    
    if len(depth_performance) == 0:
        print("No depth performance data available for visualization")
        return
    
    # Set colors from config or use defaults
    color_tiger = config['visualization']['color_tiger'] if config else "#d62728"
    color_goat = config['visualization']['color_goat'] if config else "#2ca02c"
    
    # Plot 1: Win rates by depth with confidence intervals
    plt.figure(figsize=(12, 7))
    
    # Create positions for the bars
    positions = np.arange(len(depth_performance))
    
    # Plot overall win rate bars
    bars = plt.bar(
        positions,
        depth_performance['Win Rate'],
        yerr=[
            depth_performance['Win Rate'] - depth_performance['CI Lower'],
            depth_performance['CI Upper'] - depth_performance['Win Rate']
        ],
        width=0.6,
        capsize=5,
        color='blue',
        alpha=0.7,
        label='Overall Win Rate'
    )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Set labels and title
    plt.xlabel('Minimax Depth', fontsize=14)
    plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
    plt.title('Minimax Performance by Depth', fontsize=16)
    plt.ylim(0, 1)
    
    # Set x-ticks
    plt.xticks(positions, depth_performance['Depth'], fontsize=12)
    
    # Add number of games as text
    for i, games in enumerate(depth_performance['Games']):
        plt.text(
            positions[i],
            0.05,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_win_rates.png'), dpi=300)
    plt.close()
    
    # Plot 2: Tiger vs Goat win rates by depth
    plt.figure(figsize=(12, 7))
    
    # Create positions
    positions = np.arange(len(depth_performance))
    width = 0.35
    
    # Plot bars for Tiger and Goat roles
    tiger_bars = plt.bar(
        positions - width/2,
        depth_performance['As Tiger Win Rate'],
        width=width,
        color=color_tiger,
        alpha=0.7,
        label='As Tiger'
    )
    
    goat_bars = plt.bar(
        positions + width/2,
        depth_performance['As Goat Win Rate'],
        width=width,
        color=color_goat,
        alpha=0.7,
        label='As Goat'
    )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add data labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
    add_labels(tiger_bars)
    add_labels(goat_bars)
    
    # Set labels and title
    plt.xlabel('Minimax Depth', fontsize=14)
    plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
    plt.title('Minimax Performance by Depth and Role', fontsize=16)
    plt.ylim(0, 1)
    
    # Set x-ticks
    plt.xticks(positions, depth_performance['Depth'], fontsize=12)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_role_win_rates.png'), dpi=300)
    plt.close()
    
    # Plot 3: Dual-axis plot for win rate and computation time
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot win rate on first axis
    color = 'tab:blue'
    ax1.set_xlabel('Minimax Depth', fontsize=14)
    ax1.set_ylabel('Win Rate', color=color, fontsize=14)
    
    line1 = ax1.plot(
        depth_performance['Depth'],
        depth_performance['Win Rate'],
        marker='o',
        color=color,
        linewidth=2,
        label='Win Rate'
    )
    
    # Add confidence interval shading
    ax1.fill_between(
        depth_performance['Depth'],
        depth_performance['CI Lower'],
        depth_performance['CI Upper'],
        color=color,
        alpha=0.2
    )
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)
    
    # Create second axis for computation time
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Move Time (s)', color=color, fontsize=14)
    
    line2 = ax2.plot(
        depth_performance['Depth'],
        depth_performance['Avg Move Time (s)'],
        marker='s',
        color=color,
        linewidth=2,
        label='Avg Move Time'
    )
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add a horizontal line for 50% win rate
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=12, loc='upper center')
    
    plt.title('Minimax Depth: Win Rate vs Computation Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_winrate_time.png'), dpi=300)
    plt.close()

def create_matchup_visualizations(performance_metrics, output_dir, config=None):
    """
    Create visualizations of MCTS vs Minimax matchups.
    
    Args:
        performance_metrics: Dictionary with performance metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    config_matchups = performance_metrics['config_matchups']
    mcts_configs = performance_metrics['mcts_configs']
    minimax_depths = performance_metrics['minimax_depths']
    
    # Plot 1: Heatmap of matchup win rates
    # Create a pivot table for the heatmap
    heatmap_data = config_matchups.pivot_table(
        values='Win Rate',
        index='MCTS Config',
        columns='Minimax Depth',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'MCTS Win Rate (draws = 0.5)'}
    )
    
    # Set labels and title
    plt.xlabel('Minimax Depth', fontsize=14)
    plt.ylabel('MCTS Configuration', fontsize=14)
    plt.title('MCTS vs Minimax Matchup Win Rates', fontsize=16)
    
    # Adjust font size for annotations
    for text in ax.texts:
        text.set_fontsize(9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matchup_heatmap.png'), dpi=300)
    plt.close()
    
    # Plot 2: Grouped bar chart of matchup win rates by role
    # Get unique MCTS configs and Minimax depths
    mcts_configs = sorted(config_matchups['MCTS Config'].unique())
    minimax_depths = sorted(config_matchups['Minimax Depth'].unique())
    
    # Create one plot for each MCTS config
    for mcts_config in mcts_configs:
        plt.figure(figsize=(12, 7))
        
        # Filter data for this MCTS config
        config_data = config_matchups[config_matchups['MCTS Config'] == mcts_config]
        
        # Set positions for the bars
        positions = np.arange(len(config_data))
        width = 0.35
        
        # Plot bars for Tiger and Goat roles
        tiger_bars = plt.bar(
            positions - width/2,
            config_data['As Tiger Win Rate'],
            width=width,
            color='red',
            alpha=0.7,
            label='MCTS as Tiger'
        )
        
        goat_bars = plt.bar(
            positions + width/2,
            config_data['As Goat Win Rate'],
            width=width,
            color='green',
            alpha=0.7,
            label='MCTS as Goat'
        )
        
        # Add a horizontal line for 50% win rate
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add data labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.01,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        add_labels(tiger_bars)
        add_labels(goat_bars)
        
        # Set labels and title
        plt.xlabel('Minimax Depth', fontsize=14)
        plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
        plt.title(f'{mcts_config} vs Minimax Matchups', fontsize=16)
        plt.ylim(0, 1)
        
        # Set x-ticks
        plt.xticks(positions, [f"Depth {int(d)}" for d in config_data['Minimax Depth']], fontsize=12)
        
        # Add number of games as text
        for i, row in enumerate(config_data.itertuples()):
            plt.text(
                positions[i] - width/2,
                0.05,
                f'n={row._8}',
                ha='center',
                va='bottom',
                fontsize=8
            )
            plt.text(
                positions[i] + width/2,
                0.05,
                f'n={row._12}',
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Create a safe filename
        safe_config_name = mcts_config.replace('/', '_').replace('\\', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'matchup_{safe_config_name}.png'), dpi=300)
        plt.close()
    
    # Plot 3: Line chart of MCTS performance across Minimax depths
    plt.figure(figsize=(12, 7))
    
    # Calculate average performance of each MCTS config against each Minimax depth
    avg_performance = config_matchups.groupby(['MCTS Config', 'Minimax Depth'])['Win Rate'].mean().reset_index()
    
    # Plot a line for each MCTS config
    for mcts_config in mcts_configs:
        config_data = avg_performance[avg_performance['MCTS Config'] == mcts_config]
        
        plt.plot(
            config_data['Minimax Depth'],
            config_data['Win Rate'],
            marker='o',
            linewidth=2,
            label=mcts_config
        )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    plt.xlabel('Minimax Depth', fontsize=14)
    plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
    plt.title('MCTS Performance Across Minimax Depths', fontsize=16)
    plt.ylim(0, 1)
    
    # Set x-ticks to integer values only
    plt.xticks(sorted(minimax_depths), fontsize=12)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mcts_vs_depths.png'), dpi=300)
    plt.close()

def create_capture_visualizations(game_dynamics, output_dir, config=None):
    """
    Create visualizations of capture patterns.
    
    Args:
        game_dynamics: Dictionary with game dynamics metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    capture_analysis = game_dynamics['capture_analysis']
    comeback_analysis = game_dynamics['comeback_analysis']
    
    # Plot 1: Win rates by number of captures
    plt.figure(figsize=(12, 7))
    
    # Filter to exclude cases with zero games
    filtered_capture_analysis = capture_analysis[capture_analysis['Games'] > 0]
    
    # Set positions for the bars
    positions = np.arange(len(filtered_capture_analysis))
    width = 0.3
    
    # Plot bars for Tiger and Goat win rates
    plt.bar(
        positions - width,
        filtered_capture_analysis['Tiger Win %'],
        width=width,
        color='red',
        alpha=0.7,
        label='Tiger Wins'
    )
    
    plt.bar(
        positions,
        filtered_capture_analysis['Draw %'],
        width=width,
        color='gray',
        alpha=0.7,
        label='Draws'
    )
    
    plt.bar(
        positions + width,
        filtered_capture_analysis['Goat Win %'],
        width=width,
        color='green',
        alpha=0.7,
        label='Goat Wins'
    )
    
    # Set labels and title
    plt.xlabel('Number of Captures', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.title('Game Outcomes by Number of Captures', fontsize=16)
    plt.ylim(0, 100)
    
    # Set x-ticks
    plt.xticks(positions, filtered_capture_analysis['Captures'], fontsize=12)
    
    # Add number of games as text
    for i, games in enumerate(filtered_capture_analysis['Games']):
        plt.text(
            positions[i],
            5,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'capture_outcomes.png'), dpi=300)
    plt.close()
    
    # Plot 2: Goat comeback wins
    plt.figure(figsize=(12, 7))
    
    # Filter to exclude cases with zero games
    filtered_comeback_analysis = comeback_analysis[comeback_analysis['Games'] > 0]
    
    if len(filtered_comeback_analysis) > 0:
        # Set positions for the bars
        positions = np.arange(len(filtered_comeback_analysis))
        width = 0.35
        
        # Plot bars for MCTS and Minimax
        mcts_bars = plt.bar(
            positions - width/2,
            filtered_comeback_analysis['MCTS Games'],
            width=width,
            color='blue',
            alpha=0.7,
            label='MCTS'
        )
        
        minimax_bars = plt.bar(
            positions + width/2,
            filtered_comeback_analysis['Minimax Games'],
            width=width,
            color='orange',
            alpha=0.7,
            label='Minimax'
        )
        
        # Add data labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 1,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
        
        add_labels(mcts_bars)
        add_labels(minimax_bars)
        
        # Set labels and title
        plt.xlabel('Number of Captures', fontsize=14)
        plt.ylabel('Number of Goat Wins', fontsize=14)
        plt.title('Goat "Comeback" Wins Despite Captures', fontsize=16)
        
        # Set x-ticks
        plt.xticks(positions, filtered_comeback_analysis['Captures'], fontsize=12)
        
        # Add total games as text
        for i, games in enumerate(filtered_comeback_analysis['Games']):
            plt.text(
                positions[i],
                1,
                f'Total: {games}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goat_comeback_wins.png'), dpi=300)
        plt.close()
    else:
        print("No goat comeback data available for visualization")
    
    # Plot 3: First capture move timing
    plt.figure(figsize=(12, 7))
    
    # Filter out rows with no first capture data
    first_capture_data = capture_analysis[capture_analysis['Avg First Capture Move'].notna()]
    
    if len(first_capture_data) > 0:
        # Set positions for the bars
        positions = np.arange(len(first_capture_data))
        
        # Plot bars for first capture timing
        plt.bar(
            positions,
            first_capture_data['Avg First Capture Move'],
            color='purple',
            alpha=0.7
        )
        
        # Set labels and title
        plt.xlabel('Number of Captures', fontsize=14)
        plt.ylabel('Average Move Number of First Capture', fontsize=14)
        plt.title('First Capture Timing by Final Capture Count', fontsize=16)
        
        # Set x-ticks
        plt.xticks(positions, first_capture_data['Captures'], fontsize=12)
        
        # Add data labels
        for i, avg_move in enumerate(first_capture_data['Avg First Capture Move']):
            if not pd.isna(avg_move):
                plt.text(
                    positions[i],
                    avg_move + 1,
                    f'{avg_move:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'first_capture_timing.png'), dpi=300)
        plt.close()
    else:
        print("No first capture timing data available for visualization")
    
    # Plot 4: Game length vs. captures scatter plot
    plt.figure(figsize=(12, 7))
    
    # Use sizes proportional to number of games
    sizes = capture_analysis['Games'] / capture_analysis['Games'].max() * 500
    
    # Create scatter plot
    scatter = plt.scatter(
        capture_analysis['Captures'],
        capture_analysis['Avg Game Length'],
        s=sizes,
        c=capture_analysis['Captures'],
        cmap='viridis',
        alpha=0.7
    )
    
    # Add labels for each point
    for i, row in capture_analysis.iterrows():
        plt.annotate(
            f"{int(row['Captures'])} captures\n{row['Games']} games",
            (row['Captures'], row['Avg Game Length']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )
    
    # Set labels and title
    plt.xlabel('Number of Captures', fontsize=14)
    plt.ylabel('Average Game Length (moves)', fontsize=14)
    plt.title('Relationship Between Captures and Game Length', fontsize=16)
    
    # Set x-ticks to integers
    plt.xticks(np.arange(0, capture_analysis['Captures'].max() + 1), fontsize=12)
    
    # Add colorbar for reference
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Captures')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'captures_vs_length.png'), dpi=300)
    plt.close()

def create_game_length_visualizations(game_dynamics, output_dir, config=None):
    """
    Create visualizations of game length patterns.
    
    Args:
        game_dynamics: Dictionary with game dynamics metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    length_analysis = game_dynamics['length_analysis']
    
    # Plot 1: Win rates by game length range
    plt.figure(figsize=(14, 8))
    
    # Filter to exclude cases with zero games
    filtered_length_analysis = length_analysis[length_analysis['Games'] > 0]
    
    # Set positions for the bars
    positions = np.arange(len(filtered_length_analysis))
    width = 0.3
    
    # Plot bars for Tiger and Goat win rates
    plt.bar(
        positions - width,
        filtered_length_analysis['Tiger Win %'],
        width=width,
        color='red',
        alpha=0.7,
        label='Tiger Wins'
    )
    
    plt.bar(
        positions,
        filtered_length_analysis['Draw %'],
        width=width,
        color='gray',
        alpha=0.7,
        label='Draws'
    )
    
    plt.bar(
        positions + width,
        filtered_length_analysis['Goat Win %'],
        width=width,
        color='green',
        alpha=0.7,
        label='Goat Wins'
    )
    
    # Set labels and title
    plt.xlabel('Game Length Range (moves)', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.title('Game Outcomes by Length Range', fontsize=16)
    plt.ylim(0, 100)
    
    # Set x-ticks
    plt.xticks(positions, filtered_length_analysis['Length Range'], fontsize=10, rotation=45, ha='right')
    
    # Add number of games as text
    for i, games in enumerate(filtered_length_analysis['Games']):
        plt.text(
            positions[i],
            5,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_outcomes.png'), dpi=300)
    plt.close()
    
    # Plot 2: Game length distribution by outcome
    plt.figure(figsize=(12, 7))
    
    # Calculate average game length by outcome for a bar chart
    avg_lengths = {
        'Tiger Wins': game_dynamics['avg_length_tiger_win'],
        'Goat Wins': game_dynamics['avg_length_goat_win'],
        'Draws': game_dynamics['avg_length_draw']
    }
    
    # Sort from shortest to longest
    avg_lengths = {k: v for k, v in sorted(avg_lengths.items(), key=lambda item: item[1])}
    
    # Set positions for the bars
    positions = np.arange(len(avg_lengths))
    
    # Plot bars
    bars = plt.bar(
        positions,
        avg_lengths.values(),
        color=['red', 'green', 'gray'],
        alpha=0.7
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Set labels and title
    plt.xlabel('Game Outcome', fontsize=14)
    plt.ylabel('Average Game Length (moves)', fontsize=14)
    plt.title('Average Game Length by Outcome', fontsize=16)
    
    # Set x-ticks
    plt.xticks(positions, avg_lengths.keys(), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_length_by_outcome.png'), dpi=300)
    plt.close()
    
    # Plot 3: Captures vs. Game Length correlation
    plt.figure(figsize=(12, 7))
    
    # Create a scatter plot
    plt.scatter(
        filtered_length_analysis['Avg Captures'],
        [int(r.split('-')[0]) for r in filtered_length_analysis['Length Range']],  # Use lower bound of range
        s=filtered_length_analysis['Games'] / filtered_length_analysis['Games'].max() * 300,
        c=filtered_length_analysis['Tiger Win %'],
        cmap='coolwarm',
        alpha=0.7
    )
    
    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Tiger Win %')
    
    # Set labels and title
    plt.xlabel('Average Number of Captures', fontsize=14)
    plt.ylabel('Game Length (lower bound of range)', fontsize=14)
    plt.title('Relationship Between Captures, Game Length, and Tiger Win Rate', fontsize=16)
    
    # Add a text label for the correlation coefficient
    correlation = np.corrcoef(
        filtered_length_analysis['Avg Captures'],
        [int(r.split('-')[0]) for r in filtered_length_analysis['Length Range']]
    )[0, 1]
    
    plt.text(
        0.05, 0.95,
        f'Correlation: {correlation:.3f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_captures_correlation.png'), dpi=300)
    plt.close()

def create_repetition_visualizations(game_dynamics, output_dir, config=None):
    """
    Create visualizations of threefold repetition patterns.
    
    Args:
        game_dynamics: Dictionary with game dynamics metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    repetition_analysis = game_dynamics['repetition_analysis']
    
    if len(repetition_analysis) == 0:
        print("No repetition analysis data available for visualization")
        return
    
    # Plot 1: Comparison of repetition vs. normal draws
    plt.figure(figsize=(12, 7))
    
    # Set positions for the bars
    positions = np.arange(len(repetition_analysis))
    width = 0.35
    
    # Plot bars for average length and captures
    length_bars = plt.bar(
        positions - width/2,
        repetition_analysis['Avg Length'],
        width=width,
        color='blue',
        alpha=0.7,
        label='Avg Game Length'
    )
    
    captures_bars = plt.bar(
        positions + width/2,
        repetition_analysis['Avg Captures'],
        width=width,
        color='orange',
        alpha=0.7,
        label='Avg Captures'
    )
    
    # Add data labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
    add_labels(length_bars)
    add_labels(captures_bars)
    
    # Set labels and title
    plt.xlabel('Draw Type', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Comparison of Repetition vs. Normal Draws', fontsize=16)
    
    # Set x-ticks
    plt.xticks(positions, repetition_analysis['Draw Type'], fontsize=12)
    
    # Add number of games as text
    for i, games in enumerate(repetition_analysis['Games']):
        plt.text(
            positions[i],
            1,
            f'n={games}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repetition_vs_normal.png'), dpi=300)
    plt.close()
    
    # Plot 2: Algorithm tendencies for repetition draws
    plt.figure(figsize=(14, 8))
    
    # Create data for the algorithm comparison
    algorithm_roles = ['MCTS as Tiger', 'MCTS as Goat', 'Minimax as Tiger', 'Minimax as Goat']
    repetition_counts = [
        repetition_analysis[repetition_analysis['Draw Type'] == 'Repetition']['MCTS as Tiger'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Repetition']['MCTS as Goat'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Repetition']['Minimax as Tiger'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Repetition']['Minimax as Goat'].iloc[0]
    ]
    
    normal_counts = [
        repetition_analysis[repetition_analysis['Draw Type'] == 'Normal']['MCTS as Tiger'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Normal']['MCTS as Goat'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Normal']['Minimax as Tiger'].iloc[0],
        repetition_analysis[repetition_analysis['Draw Type'] == 'Normal']['Minimax as Goat'].iloc[0]
    ]
    
    # Calculate total draws for each algorithm role
    total_counts = [r + n for r, n in zip(repetition_counts, normal_counts)]
    
    # Calculate percentages
    repetition_pcts = [r / t * 100 if t > 0 else 0 for r, t in zip(repetition_counts, total_counts)]
    normal_pcts = [n / t * 100 if t > 0 else 0 for n, t in zip(normal_counts, total_counts)]
    
    # Set positions for the bars
    positions = np.arange(len(algorithm_roles))
    width = 0.35
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Counts
    ax1.bar(positions - width/2, repetition_counts, width, label='Repetition Draws', color='purple', alpha=0.7)
    ax1.bar(positions + width/2, normal_counts, width, label='Normal Draws', color='gray', alpha=0.7)
    
    ax1.set_xlabel('Algorithm Role', fontsize=14)
    ax1.set_ylabel('Number of Games', fontsize=14)
    ax1.set_title('Draw Counts by Algorithm and Role', fontsize=16)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(algorithm_roles, fontsize=10, rotation=45, ha='right')
    ax1.legend(fontsize=12)
    
    # Add counts as labels
    for i, count in enumerate(repetition_counts):
        ax1.text(positions[i] - width/2, count + 1, str(count), ha='center', va='bottom', fontsize=9)
    for i, count in enumerate(normal_counts):
        ax1.text(positions[i] + width/2, count + 1, str(count), ha='center', va='bottom', fontsize=9)
    
    # Right subplot: Percentages
    ax2.bar(positions - width/2, repetition_pcts, width, label='Repetition Draws %', color='purple', alpha=0.7)
    ax2.bar(positions + width/2, normal_pcts, width, label='Normal Draws %', color='gray', alpha=0.7)
    
    ax2.set_xlabel('Algorithm Role', fontsize=14)
    ax2.set_ylabel('Percentage of Draws', fontsize=14)
    ax2.set_title('Draw Percentages by Algorithm and Role', fontsize=16)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(algorithm_roles, fontsize=10, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=12)
    
    # Add percentages as labels
    for i, pct in enumerate(repetition_pcts):
        ax2.text(positions[i] - width/2, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, pct in enumerate(normal_pcts):
        ax2.text(positions[i] + width/2, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Add total count information as text
    for i, total in enumerate(total_counts):
        ax2.text(positions[i], 90, f'Total: {total}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_draw_tendencies.png'), dpi=300)
    plt.close()

def create_movement_visualizations(movement_patterns, output_dir, config=None):
    """
    Create visualizations of movement patterns.
    
    Args:
        movement_patterns: Dictionary with movement pattern metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    opening_analysis = movement_patterns['opening_analysis']
    capture_pattern_analysis = movement_patterns['capture_pattern_analysis']
    draw_pattern_analysis = movement_patterns['draw_pattern_analysis']
    
    # Plot 1: Tiger response success rates
    if len(opening_analysis) > 0:
        # Group by opening
        openings = opening_analysis['Opening'].unique()
        
        for opening in openings:
            opening_data = opening_analysis[opening_analysis['Opening'] == opening]
            
            if len(opening_data) > 0:
                plt.figure(figsize=(14, 8))
                
                # Sort by number of games
                opening_data = opening_data.sort_values('Games', ascending=False)
                
                # Set positions for the bars
                positions = np.arange(len(opening_data))
                width = 0.3
                
                # Plot bars for win rates
                plt.bar(
                    positions - width,
                    opening_data['Tiger Win %'],
                    width=width,
                    color='red',
                    alpha=0.7,
                    label='Tiger Win %'
                )
                
                plt.bar(
                    positions,
                    opening_data['Draw %'],
                    width=width,
                    color='gray',
                    alpha=0.7,
                    label='Draw %'
                )
                
                plt.bar(
                    positions + width,
                    opening_data['Goat Win %'],
                    width=width,
                    color='green',
                    alpha=0.7,
                    label='Goat Win %'
                )
                
                # Set labels and title
                plt.xlabel('Tiger Response', fontsize=14)
                plt.ylabel('Win Rate (%)', fontsize=14)
                plt.title(f'Success Rates of Tiger Responses to {opening}', fontsize=16)
                plt.ylim(0, 100)
                
                # Set x-ticks
                plt.xticks(positions, opening_data['Response'], fontsize=10, rotation=45, ha='right')
                
                # Add number of games as text
                for i, games in enumerate(opening_data['Games']):
                    plt.text(
                        positions[i],
                        5,
                        f'n={games}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )
                
                plt.legend(fontsize=12)
                plt.tight_layout()
                
                # Create a safe filename
                safe_opening_name = opening.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('@', 'at')
                plt.savefig(os.path.join(output_dir, f'response_{safe_opening_name}.png'), dpi=300)
                plt.close()
    else:
        print("No opening analysis data available for visualization")
    
    # Plot 2: Capture positions heatmap
    if len(capture_pattern_analysis) > 0:
        plt.figure(figsize=(10, 8))
        
        # Create a 5x5 grid for the board
        capture_grid = np.zeros((5, 5))
        
        # Fill in the capture counts
        for _, row in capture_pattern_analysis.iterrows():
            capture_grid[row['Y'], row['X']] = row['Count']
        
        # Create heatmap
        sns.heatmap(
            capture_grid,
            annot=True,
            fmt='.0f',
            cmap='Reds',
            cbar_kws={'label': 'Number of Captures'}
        )
        
        # Set labels and title
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.title('Spatial Distribution of Captures', fontsize=16)
        
        # Set axis labels to match board coordinates
        plt.xticks(np.arange(5) + 0.5, np.arange(5))
        plt.yticks(np.arange(5) + 0.5, np.arange(5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'capture_positions.png'), dpi=300)
        plt.close()
        
        # Plot 3: Capture positions on a board visualization
        plt.figure(figsize=(10, 10))
        
        # Draw the board
        ax = plt.gca()
        
        # Draw the grid
        for i in range(5):
            plt.axhline(y=i, color='black', linewidth=1)
            plt.axvline(x=i, color='black', linewidth=1)
        
        # Draw the diagonals for the center and corners
        plt.plot([0, 4], [0, 4], 'k-', linewidth=1)  # Diagonal from top-left to bottom-right
        plt.plot([0, 4], [4, 0], 'k-', linewidth=1)  # Diagonal from bottom-left to top-right
        
        # Add middle-edge diagonals
        plt.plot([1, 3], [1, 3], 'k-', linewidth=1)
        plt.plot([1, 3], [3, 1], 'k-', linewidth=1)
        
        # Normalize the capture counts for sizing the circles
        max_count = capture_pattern_analysis['Count'].max()
        norm_counts = capture_pattern_analysis['Count'] / max_count
        
        # Plot circles at capture positions, sized by frequency
        for _, row in capture_pattern_analysis.iterrows():
            size = 1000 * (row['Count'] / max_count)  # Scale circle size
            plt.scatter(row['X'], row['Y'], s=size, color='red', alpha=0.6)
            plt.text(row['X'], row['Y'], str(int(row['Count'])), ha='center', va='center', fontsize=9)
        
        # Set limits and aspect ratio
        plt.xlim(-0.5, 4.5)
        plt.ylim(-0.5, 4.5)
        plt.gca().set_aspect('equal')
        
        # Set labels and title
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.title('Capture Positions on Board', fontsize=16)
        
        # Invert y-axis to match typical board orientation
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'capture_board.png'), dpi=300)
        plt.close()
    else:
        print("No capture pattern analysis data available for visualization")
    
    # Plot 4: Draw position analysis
    if len(draw_pattern_analysis) > 0:
        plt.figure(figsize=(12, 7))
        
        # Sort by count
        draw_pattern_analysis = draw_pattern_analysis.sort_values('Count', ascending=False)
        
        # Set positions for the bars
        positions = np.arange(len(draw_pattern_analysis))
        
        # Plot bars
        plt.bar(
            positions,
            draw_pattern_analysis['Count'],
            color='purple',
            alpha=0.7
        )
        
        # Set labels and title
        plt.xlabel('Board Position', fontsize=14)
        plt.ylabel('Number of Games', fontsize=14)
        plt.title('Most Common Final Positions in Draw Games', fontsize=16)
        
        # Set x-ticks
        plt.xticks(positions, draw_pattern_analysis['Position'], fontsize=10, rotation=45, ha='right')
        
        # Add percentages as text
        for i, pct in enumerate(draw_pattern_analysis['Percentage']):
            plt.text(
                positions[i],
                draw_pattern_analysis['Count'].iloc[i] + 1,
                f'{pct:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'draw_positions.png'), dpi=300)
        plt.close()
        
        # Plot 5: Draw positions on a board visualization
        plt.figure(figsize=(10, 10))
        
        # Draw the board
        ax = plt.gca()
        
        # Draw the grid
        for i in range(5):
            plt.axhline(y=i, color='black', linewidth=1)
            plt.axvline(x=i, color='black', linewidth=1)
        
        # Draw the diagonals for the center and corners
        plt.plot([0, 4], [0, 4], 'k-', linewidth=1)  # Diagonal from top-left to bottom-right
        plt.plot([0, 4], [4, 0], 'k-', linewidth=1)  # Diagonal from bottom-left to top-right
        
        # Add middle-edge diagonals
        plt.plot([1, 3], [1, 3], 'k-', linewidth=1)
        plt.plot([1, 3], [3, 1], 'k-', linewidth=1)
        
        # Normalize the counts for sizing the circles
        max_count = draw_pattern_analysis['Count'].max()
        
        # Plot circles at draw positions, sized by frequency
        for _, row in draw_pattern_analysis.iterrows():
            size = 1000 * (row['Count'] / max_count)  # Scale circle size
            plt.scatter(row['X'], row['Y'], s=size, color='purple', alpha=0.6)
            plt.text(row['X'], row['Y'], str(int(row['Count'])), ha='center', va='center', fontsize=9)
        
        # Set limits and aspect ratio
        plt.xlim(-0.5, 4.5)
        plt.ylim(-0.5, 4.5)
        plt.gca().set_aspect('equal')
        
        # Set labels and title
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.title('Common Final Positions in Draw Games', fontsize=16)
        
        # Invert y-axis to match typical board orientation
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'draw_board.png'), dpi=300)
        plt.close()
    else:
        print("No draw pattern analysis data available for visualization") 