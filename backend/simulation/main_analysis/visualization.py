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
    
    # Add number of games as text above bars (win rate vis)
    for i, games in enumerate(algorithm_comparison['Games']):
        plt.text(
            positions[i],
            5,
            str(games),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_win_rates.png'), dpi=300)
    plt.close()
    
    # Plot 2: Win Rates by Role
    plt.figure(figsize=(12, 7))
    
    # Create bar positions
    positions = np.arange(len(role_performance))
    bar_width = 0.25
    
    # Create bars for each outcome with percentages (multiply by 100)
    plt.bar(
        positions - bar_width,
        role_performance['Win %'] * 100,
        width=bar_width,
        color='green',
        alpha=0.7,
        label='Win'
    )
    
    plt.bar(
        positions,
        role_performance['Draw %'] * 100,
        width=bar_width,
        color='gray',
        alpha=0.7,
        label='Draw'
    )
    
    plt.bar(
        positions + bar_width,
        role_performance['Loss %'] * 100,
        width=bar_width,
        color='red',
        alpha=0.7,
        label='Loss'
    )
    
    # Add error bars for adjusted win rate (multiply by 100 for percentage)
    plt.errorbar(
        positions,
        role_performance['Win Rate'] * 100,
        yerr=[
            (role_performance['Win Rate'] - role_performance['CI Lower']) * 100,
            (role_performance['CI Upper'] - role_performance['Win Rate']) * 100
        ],
        fmt='o',
        color='black',
        capsize=5,
        label='Adjusted Win Rate (with 95% CI)'
    )
    
    # Add a horizontal line for 50% win rate
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Set labels and title
    plt.xlabel('Role', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title('Performance by Algorithm and Role', fontsize=16)
    plt.ylim(0, 100)
    
    # Set x-ticks
    plt.xticks(positions, role_performance['Role'], fontsize=12)
    
    # Add number of games as text above bars (win rate vis)
    for i, games in enumerate(role_performance['Games']):
        plt.text(
            positions[i],
            5,
            str(games),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
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
    
    # Add number of games as text above bars (win rate vis)
    for i, games in enumerate(depth_performance['Games']):
        plt.text(
            positions[i],
            5,
            str(games),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
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
    Create visualizations of algorithm matchups.
    
    Args:
        performance_metrics: Dictionary with performance metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    config_matchups = performance_metrics['config_matchups']
    mcts_configs = performance_metrics['mcts_configs']
    
    # Set colors from config or use defaults
    color_tiger = config['visualization']['color_tiger'] if config else "#d62728"
    color_goat = config['visualization']['color_goat'] if config else "#2ca02c"
    
    # Get unique MCTS configurations
    unique_mcts_configs = sorted(list(set(config_matchups['MCTS Config'])))
    
    # Create a plot for each MCTS configuration
    for mcts_config in unique_mcts_configs:
        # Filter matchups for this MCTS configuration
        config_data = config_matchups[config_matchups['MCTS Config'] == mcts_config]
        
        if len(config_data) > 0:
            plt.figure(figsize=(12, 8))
            
            # Sort by depth
            config_data = config_data.sort_values('Minimax Depth')
            
            # Set positions for the bars
            positions = np.arange(len(config_data))
            width = 0.35  # Slightly increased bar width
            
            # Create bars for MCTS as Tiger and MCTS as Goat
            plt.bar(
                positions - width/2,
                config_data['As Tiger Win Rate'],
                width=width,
                color=color_tiger,
                alpha=0.7,
                label='MCTS as Tiger'
            )
            
            plt.bar(
                positions + width/2,
                config_data['As Goat Win Rate'],
                width=width,
                color=color_goat,
                alpha=0.7,
                label='MCTS as Goat'
            )
            
            # Add a horizontal line for 50% win rate
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
            
            # Add data labels
            def add_labels(positions, values, offset, color):
                for i, value in enumerate(values):
                    plt.text(
                        positions[i] + offset,
                        value + 0.02,
                        f'{value:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color=color
                    )
            
            add_labels(positions, config_data['As Tiger Win Rate'], -width/2, color_tiger)
            add_labels(positions, config_data['As Goat Win Rate'], width/2, color_goat)
            
            # Set labels and title
            plt.xlabel('Minimax Depth', fontsize=14)
            plt.ylabel('Win Rate (draws = 0.5)', fontsize=14)
            plt.title(f'MCTS {mcts_config} vs. Minimax', fontsize=16)
            plt.ylim(0, 1)
            
            # Set x-ticks
            plt.xticks(positions, config_data['Minimax Depth'])
            
            # Add number of games as text above bars (win rate vis)
            for i, (tiger_games, goat_games) in enumerate(zip(config_data['As Tiger Games'], config_data['As Goat Games'])):
                # Show total games in bold
                total_games = tiger_games + goat_games
                plt.text(
                    positions[i],
                    5,
                    str(total_games),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )
            
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'matchup_{mcts_config.replace("/", "_")}.png'), dpi=300)
            plt.close()

def create_capture_visualizations(game_dynamics, output_dir, config=None):
    """
    Create visualizations of capture statistics.
    
    Args:
        game_dynamics: Dictionary with game dynamics metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    capture_analysis = game_dynamics['capture_analysis']
    
    # Set colors from config or use defaults
    color_tiger = config['visualization']['color_tiger'] if config else "#d62728"
    color_goat = config['visualization']['color_goat'] if config else "#2ca02c"
    
    # Plot 1: Capture outcomes
    plt.figure(figsize=(12, 7))
    
    # Group data by captures
    x = capture_analysis['Captures']
    
    # Create positions for the bars
    positions = np.arange(len(capture_analysis))
    width = 0.25  # Increased bar width
    
    # Create bars
    plt.bar(
        positions - width,
        capture_analysis['Tiger Win %'],
        width=width,
        color=color_tiger,
        alpha=0.7,
        label='Tiger Win %'
    )
    
    plt.bar(
        positions,
        capture_analysis['Draw %'],
        width=width,
        color='gray',
        alpha=0.7,
        label='Draw %'
    )
    
    plt.bar(
        positions + width,
        capture_analysis['Goat Win %'],
        width=width,
        color=color_goat,
        alpha=0.7,
        label='Goat Win %'
    )
    
    # Set labels and title
    plt.xlabel('Number of Captures', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.title('Win Rates by Number of Captures', fontsize=16)
    
    # Set x-ticks normally (no extra point)
    plt.xticks(positions, capture_analysis['Captures'])
    
    # Add number of games as text above bars (win rate vis)
    for i, games in enumerate(capture_analysis['Games']):
        plt.text(
            positions[i],
            5,
            str(games),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Create legend with horizontal layout (ncol=3)
    plt.legend(fontsize=12, loc='upper center', ncol=3, framealpha=0.7)
    plt.ylim(0, 110)  # Extend y-axis to 110% to provide margin above 100%
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'capture_outcomes.png'), dpi=300)
    plt.close()
    
    # Plot 2: First capture timing vs. outcome
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
        plt.title('First Capture Timing by Final Capture Count', fontsize=16, pad=15)
        
        # Set x-ticks
        plt.xticks(positions, first_capture_data['Captures'], fontsize=12)
        
        # Calculate y-axis max as next multiple of 5 above the maximum value
        max_value = first_capture_data['Avg First Capture Move'].max()
        y_max = 5 * (int(max_value / 5) + 2)  # Next highest multiple of 5, plus one extra tick
        plt.ylim(0, y_max)
        
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
    
    # Plot 3: Scatter plot of capture count vs. game length
    plt.figure(figsize=(12, 8))
    
    # Create positions for the scatter plot
    x = capture_analysis['Captures']
    y = capture_analysis['Avg Game Length']
    sizes = capture_analysis['Games'] / max(capture_analysis['Games']) * 500
    
    # Create scatter plot with a single color instead of a complex color calculation
    plt.scatter(x, y, s=sizes, color='purple', alpha=0.7)
    
    # Add data labels with position based on x-value
    for i, row in capture_analysis.iterrows():
        # Position text to the right for captures < 3, to the left for captures >= 3
        ha_align = 'left' if row['Captures'] < 3 else 'right'
        x_offset = 0.2 if row['Captures'] < 3 else -0.2
        
        plt.text(
            row['Captures'] + x_offset,
            row['Avg Game Length'],
            f"{row['Captures']} captures\n{row['Games']} games\nTiger: {row['Tiger Win %']:.1f}%\nGoat: {row['Goat Win %']:.1f}%",
            ha=ha_align,
            va='center',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
        )
    
    # Set labels and title
    plt.xlabel('Number of Captures', fontsize=14)
    plt.ylabel('Average Game Length (moves)', fontsize=14)
    plt.title('Captures vs. Game Length', fontsize=16)
    
    # Set the x limits to include some padding
    plt.xlim(-0.5, max(x) + 0.8)
    
    # Create a legend for the bubble size
    max_games = max(capture_analysis['Games'])
    legend_sizes = [int(max_games/3), int(2*max_games/3), max_games]
    legend_labels = [f"{size} games" for size in legend_sizes]
    
    # Create legend elements
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor='purple', markersize=np.sqrt(size/max_games*30)) 
                   for size, label in zip(legend_sizes, legend_labels)]
    
    # Add a legend for the bubble size
    plt.legend(handles=legend_elements, title="Bubble Size", loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'captures_vs_length.png'), dpi=300)
    plt.close()
    
    # Plot 4: Goat comeback win analysis
    plt.figure(figsize=(12, 7))
    
    # Extract data
    comeback_analysis = game_dynamics['comeback_analysis']
    
    if len(comeback_analysis) > 0:
        # Group by captures
        x = comeback_analysis['Captures']
        
        # Create positions for the bars
        positions = np.arange(len(comeback_analysis))
        width = 0.35
        
        # Create bars
        mcts_bars = plt.bar(
            positions - width/2,
            comeback_analysis['MCTS Games'],
            width=width,
            color='blue',
            alpha=0.7,
            label='MCTS as Goat'
        )
        
        minimax_bars = plt.bar(
            positions + width/2,
            comeback_analysis['Minimax Games'],
            width=width,
            color='orange',
            alpha=0.7,
            label='Minimax as Goat'
        )
        
        # Add data labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 0.5,
                        str(int(height)),
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )
        
        add_labels(mcts_bars)
        add_labels(minimax_bars)
        
        # Set labels and title
        plt.xlabel('Number of Captures', fontsize=14)
        plt.ylabel('Number of Comeback Wins', fontsize=14)
        plt.title('Goat Comeback Wins Despite Captures', fontsize=16)
        plt.xticks(positions, comeback_analysis['Captures'])
        
        plt.legend(fontsize=12)
        
        # Set y-limit to allow space for labels
        max_value = max(max(comeback_analysis['MCTS Games']), max(comeback_analysis['Minimax Games']))
        plt.ylim(0, max_value * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goat_comeback_wins.png'), dpi=300)
    plt.close()

def create_game_length_visualizations(game_dynamics, output_dir, config=None):
    """
    Create visualizations of game length statistics.
    
    Args:
        game_dynamics: Dictionary with game dynamics metrics
        output_dir: Directory to save output figures
        config: Configuration dictionary
    """
    ensure_directory(output_dir)
    
    # Extract data
    length_analysis = game_dynamics['length_analysis']
    
    # Set colors from config or use defaults
    color_tiger = config['visualization']['color_tiger'] if config else "#d62728"
    color_goat = config['visualization']['color_goat'] if config else "#2ca02c"
    
    # Plot: Game outcomes by length
    plt.figure(figsize=(14, 8))
    
    # Set positions for the bars
    positions = np.arange(len(length_analysis))
    width = 0.25  # Increased bar width
    
    # Create bars
    plt.bar(
        positions - width,
        length_analysis['Tiger Win %'],
        width=width,
        color=color_tiger,
        alpha=0.7,
        label='Tiger Win %'
    )
    
    plt.bar(
        positions,
        length_analysis['Draw %'],
        width=width,
        color='gray',
        alpha=0.7,
        label='Draw %'
    )
    
    plt.bar(
        positions + width,
        length_analysis['Goat Win %'],
        width=width,
        color=color_goat,
        alpha=0.7,
        label='Goat Win %'
    )
    
    # Set labels and title
    plt.xlabel('Game Length (moves)', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.title('Game Outcomes by Length', fontsize=16)
    plt.xticks(positions, length_analysis['Length Range'], fontsize=10, rotation=45, ha='right')
    
    # Add number of games as text above bars (win rate vis)
    for i, games in enumerate(length_analysis['Games']):
        plt.text(
            positions[i],
            5,
            str(games),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add secondary axis for average captures
    ax2 = plt.twinx()
    ax2.plot(
        positions,
        length_analysis['Avg Captures'],
        'o-',
        color='purple',
        linewidth=2,
        markersize=8,
        alpha=0.7,
        label='Avg Captures'
    )
    
    # Set secondary axis label
    ax2.set_ylabel('Average Captures', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Set ylim for percentages
    plt.ylim(0, 100)
    
    # Set ylim for captures
    max_captures = max(length_analysis['Avg Captures'])
    ax2.set_ylim(0, max_captures * 1.2)
    
    # Add data labels for average captures
    for i, captures in enumerate(length_analysis['Avg Captures']):
        ax2.text(
            positions[i],
            captures + 0.1,
            f'{captures:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
            color='purple'
        )
    
    # Create combined legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add more space for the game count labels
    plt.savefig(os.path.join(output_dir, 'length_outcomes.png'), dpi=300)
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
    
    # Plot: Algorithm tendencies for threefold repetition draws
    plt.figure(figsize=(10, 7))
    
    # Create simplified data for the algorithm comparison (just two categories)
    algorithm_roles = ['MCTS as Tiger/Minimax as Goat', 'MCTS as Goat/Minimax as Tiger']
    repetition_counts = [
        repetition_analysis['MCTS as Tiger'].iloc[0],  # Same as Minimax as Goat
        repetition_analysis['MCTS as Goat'].iloc[0]    # Same as Minimax as Tiger
    ]
    
    # Set positions for the bars
    positions = np.arange(len(algorithm_roles))
    
    # Create the plot with wider bars
    bars = plt.bar(positions, repetition_counts, color='purple', alpha=0.7, width=0.5)
    
    # Add labels on top of bars
    for i, count in enumerate(repetition_counts):
        plt.text(positions[i], count + 0.5, str(count), ha='center', va='bottom', fontsize=11)
    
    # Set labels and title with more padding
    plt.xlabel('Algorithm and Role', fontsize=14, labelpad=10)
    plt.ylabel('Number of Threefold Repetition Draws', fontsize=14)
    plt.title('Threefold Repetition Draws by Algorithm and Role', fontsize=16, pad=15)
    plt.xticks(positions, algorithm_roles, fontsize=12)
    
    # Add average game length and captures as text with more bottom padding
    avg_length = repetition_analysis['Avg Length'].iloc[0]
    avg_captures = repetition_analysis['Avg Captures'].iloc[0]
    plt.figtext(0.5, 0.02, 
                f"Average Game Length: {avg_length:.1f} moves | Average Captures: {avg_captures:.1f}",
                ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add more space at the bottom
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
                plt.xticks(positions, opening_data['Response'], fontsize=10, fontweight='bold')
                
                # Add number of games as text above bars (win rate vis)
                for i, games in enumerate(opening_data['Games']):
                    plt.text(
                        positions[i],
                        5,
                        str(games),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        fontweight='bold'
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
            cbar_kws={'label': 'Number of Captures'},
            linewidths=1,
            linecolor='black'
        )
        
        # Set labels and title
        plt.title('Spatial Distribution of Captures', fontsize=16)
        
        # Set axis labels to match board coordinates
        plt.xticks(np.arange(5) + 0.5, np.arange(5))
        plt.yticks(np.arange(5) + 0.5, np.arange(5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'capture_positions.png'), dpi=300)
        plt.close()
    else:
        print("No capture pattern analysis data available for visualization") 