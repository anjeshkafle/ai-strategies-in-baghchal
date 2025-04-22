"""
Main script for MCTS vs Minimax competition analysis.
"""
import os
import pandas as pd
from . import competition_analysis
from . import visualization

def run_analysis(competition_file, output_dir, config_file=None):
    """
    Run complete analysis pipeline on competition results.
    
    Args:
        competition_file: Path to competition CSV file
        output_dir: Directory for output files
        config_file: Path to configuration JSON file
    
    Returns:
        Dictionary with analysis results
    """
    # Load configuration
    config = competition_analysis.load_config(config_file)
    
    # Create output directories
    data_dir = os.path.join(output_dir, 'data')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading competition data from {competition_file}...")
    
    # Load and process data
    competition_data = competition_analysis.load_and_preprocess_data(competition_file)
    
    print(f"Analyzing {len(competition_data)} games between MCTS and Minimax configurations...")
    
    # Perform performance analysis
    performance_metrics = competition_analysis.analyze_performance(competition_data, config)
    print("Performance analysis completed")
    
    # Perform game dynamics analysis
    game_dynamics = competition_analysis.analyze_game_dynamics(competition_data)
    print("Game dynamics analysis completed")
    
    # Perform movement pattern analysis
    movement_patterns = competition_analysis.analyze_movement_patterns(competition_data)
    print("Movement pattern analysis completed")
    
    # Perform statistical tests
    statistical_results = competition_analysis.perform_statistical_tests(competition_data)
    print("Statistical tests completed")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualization.create_win_rate_visualizations(performance_metrics, figures_dir, config)
    visualization.create_depth_performance_visualizations(performance_metrics, figures_dir, config)
    visualization.create_matchup_visualizations(performance_metrics, figures_dir, config)
    visualization.create_capture_visualizations(game_dynamics, figures_dir, config)
    visualization.create_game_length_visualizations(game_dynamics, figures_dir, config)
    visualization.create_repetition_visualizations(game_dynamics, figures_dir, config)
    visualization.create_movement_visualizations(movement_patterns, figures_dir, config)
    
    # Save processed data
    print("\nSaving processed data...")
    performance_metrics["algorithm_comparison"].to_csv(os.path.join(data_dir, "algorithm_comparison.csv"), index=False)
    performance_metrics["role_performance"].to_csv(os.path.join(data_dir, "role_performance.csv"), index=False)
    performance_metrics["depth_performance"].to_csv(os.path.join(data_dir, "depth_performance.csv"), index=False)
    performance_metrics["config_matchups"].to_csv(os.path.join(data_dir, "config_matchups.csv"), index=False)
    game_dynamics["capture_analysis"].to_csv(os.path.join(data_dir, "capture_analysis.csv"), index=False)
    game_dynamics["comeback_analysis"].to_csv(os.path.join(data_dir, "comeback_analysis.csv"), index=False)
    game_dynamics["length_analysis"].to_csv(os.path.join(data_dir, "length_analysis.csv"), index=False)
    game_dynamics["repetition_analysis"].to_csv(os.path.join(data_dir, "repetition_analysis.csv"), index=False)
    game_dynamics["time_analysis"].to_csv(os.path.join(data_dir, "time_analysis.csv"), index=False)
    movement_patterns["opening_analysis"].to_csv(os.path.join(data_dir, "opening_analysis.csv"), index=False)
    movement_patterns["capture_pattern_analysis"].to_csv(os.path.join(data_dir, "capture_pattern_analysis.csv"), index=False)
    
    # Generate summary report
    with open(os.path.join(output_dir, "competition_analysis_summary.txt"), "w") as f:
        f.write("MCTS vs Minimax Competition Analysis Summary\n")
        f.write("=========================================\n\n")
        
        f.write(f"Competition data: {competition_file}\n")
        f.write(f"Games analyzed: {len(competition_data)}\n")
        f.write(f"MCTS configurations: {', '.join(sorted(performance_metrics['mcts_configs']))}\n")
        f.write(f"Minimax depths: {', '.join(map(str, sorted(performance_metrics['minimax_depths'])))}\n\n")
        
        f.write("Performance Summary:\n")
        f.write("-----------------\n")
        f.write(f"Overall MCTS win rate: {performance_metrics['mcts_overall_win_rate']:.4f}\n")
        f.write(f"Overall Minimax win rate: {performance_metrics['minimax_overall_win_rate']:.4f}\n")
        f.write(f"Draw rate: {performance_metrics['draw_rate']:.4f}\n\n")
        
        f.write("Win Rates by Role:\n")
        f.write("---------------\n")
        f.write(f"MCTS as Tiger win rate: {performance_metrics['mcts_as_tiger_win_rate']:.4f}\n")
        f.write(f"MCTS as Goat win rate: {performance_metrics['mcts_as_goat_win_rate']:.4f}\n")
        f.write(f"Minimax as Tiger win rate: {performance_metrics['minimax_as_tiger_win_rate']:.4f}\n")
        f.write(f"Minimax as Goat win rate: {performance_metrics['minimax_as_goat_win_rate']:.4f}\n\n")
        
        f.write("Best Performing Configurations:\n")
        f.write("--------------------------\n")
        
        # Check if we're showing all configurations
        if config.get('top_configs', {}).get('count', 3) <= 0:
            f.write("(Showing all configurations)\n")
            
        f.write("MCTS:\n")
        for i, config in enumerate(performance_metrics['top_mcts_configs']):
            f.write(f"{i+1}. {config['config_id']} - Win rate: {config['win_rate']:.4f}\n")
        
        f.write("\nMinimax:\n")
        for i, config in enumerate(performance_metrics['top_minimax_configs']):
            f.write(f"{i+1}. Depth {config['depth']} - Win rate: {config['win_rate']:.4f}\n\n")
        
        f.write("Game Dynamics Summary:\n")
        f.write("-------------------\n")
        f.write(f"Average game length: {game_dynamics['avg_game_length']:.2f} moves\n")
        f.write(f"Average game length by winner:\n")
        f.write(f"  Tiger wins: {game_dynamics['avg_length_tiger_win']:.2f} moves\n")
        f.write(f"  Goat wins: {game_dynamics['avg_length_goat_win']:.2f} moves\n")
        f.write(f"  Draws: {game_dynamics['avg_length_draw']:.2f} moves\n\n")
        
        f.write(f"Goat comeback wins (despite captures):\n")
        f.write(f"  1 capture: {game_dynamics['goat_wins_1_capture']} games\n")
        f.write(f"  2 captures: {game_dynamics['goat_wins_2_capture']} games\n")
        f.write(f"  3 captures: {game_dynamics['goat_wins_3_capture']} games\n")
        f.write(f"  4 captures: {game_dynamics['goat_wins_4_capture']} games\n\n")
        
        f.write("Statistical Results:\n")
        f.write("-----------------\n")
        significance_threshold = config.get('statistical', {}).get('significance_threshold', 0.05)
        
        # Add statistical results
        if 'algorithm_comparison_test' in statistical_results:
            test = statistical_results['algorithm_comparison_test']
            is_significant = test['p_value'] < significance_threshold
            significance = "SIGNIFICANT" if is_significant else "NOT significant"
            f.write(f"Algorithm Performance T-Test: t={test['statistic']:.4f}, p={test['p_value']:.4f} ({significance})\n")
            f.write(f"  This means the difference in win rates between MCTS and Minimax is {significance.lower()}.\n\n")
        
        if 'depth_comparison_tests' in statistical_results:
            f.write("Minimax Depth Comparison Tests:\n")
            for key, test in statistical_results['depth_comparison_tests'].items():
                is_significant = test['p_value'] < significance_threshold
                significance = "SIGNIFICANT" if is_significant else "NOT significant"
                d1, d2 = key.split('_vs_')
                f.write(f"  Depth {d1} vs Depth {d2}: t={test['statistic']:.4f}, p={test['p_value']:.4f} ({significance})\n")
            f.write("\n")
        
        f.write("Analysis Outputs:\n")
        f.write("---------------\n")
        f.write(f"Data files in: {data_dir}\n")
        f.write(f"Visualizations in: {figures_dir}\n")
    
    print(f"\nAnalysis summary saved to {os.path.join(output_dir, 'competition_analysis_summary.txt')}")
    
    return {
        'performance_metrics': performance_metrics,
        'game_dynamics': game_dynamics,
        'movement_patterns': movement_patterns,
        'statistical_results': statistical_results
    } 