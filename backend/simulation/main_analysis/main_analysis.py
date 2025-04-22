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
    
    # Generate detailed statistical validation report
    statistical_report = competition_analysis.generate_statistical_report(statistical_results, output_dir, config)
    print(f"Statistical validation report generated: {statistical_report}")
    
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
    performance_df = pd.DataFrame(performance_metrics['algorithm_comparison'])
    performance_df.to_csv(os.path.join(data_dir, "algorithm_performance.csv"), index=False)
    
    role_df = pd.DataFrame(performance_metrics['role_performance'])
    role_df.to_csv(os.path.join(data_dir, "role_performance.csv"), index=False)
    
    depth_df = pd.DataFrame(performance_metrics['depth_performance'])
    depth_df.to_csv(os.path.join(data_dir, "depth_performance.csv"), index=False)
    
    # Restore other CSV exports that were accidentally removed
    if 'config_matchups' in performance_metrics:
        performance_metrics['config_matchups'].to_csv(os.path.join(data_dir, "config_matchups.csv"), index=False)
    
    if 'capture_analysis' in game_dynamics:
        game_dynamics['capture_analysis'].to_csv(os.path.join(data_dir, "capture_analysis.csv"), index=False)
    
    if 'comeback_analysis' in game_dynamics:
        game_dynamics['comeback_analysis'].to_csv(os.path.join(data_dir, "comeback_analysis.csv"), index=False)
    
    if 'length_analysis' in game_dynamics:
        game_dynamics['length_analysis'].to_csv(os.path.join(data_dir, "length_analysis.csv"), index=False)
    
    if 'repetition_analysis' in game_dynamics:
        game_dynamics['repetition_analysis'].to_csv(os.path.join(data_dir, "repetition_analysis.csv"), index=False)
    
    if 'time_analysis' in game_dynamics:
        game_dynamics['time_analysis'].to_csv(os.path.join(data_dir, "time_analysis.csv"), index=False)
    
    if 'opening_analysis' in movement_patterns:
        movement_patterns['opening_analysis'].to_csv(os.path.join(data_dir, "opening_analysis.csv"), index=False)
    
    if 'capture_pattern_analysis' in movement_patterns:
        movement_patterns['capture_pattern_analysis'].to_csv(os.path.join(data_dir, "capture_pattern_analysis.csv"), index=False)
    
    # Generate a summary report
    with open(os.path.join(output_dir, "competition_analysis_summary.txt"), "w") as f:
        f.write("MCTS vs Minimax Competition Analysis Summary\n")
        f.write("==========================================\n\n")
        
        f.write(f"Competition data: {competition_file}\n")
        f.write(f"Games analyzed: {len(competition_data)}\n\n")
        
        f.write("Algorithm Performance Summary:\n")
        f.write("---------------------------\n")
        for i, row in performance_metrics['algorithm_comparison'].iterrows():
            f.write(f"{row['Algorithm']}: Win Rate = {row['Win Rate']:.4f} (95% CI: [{row['CI Lower']:.4f}, {row['CI Upper']:.4f}])\n")
            f.write(f"  Games played: {row['Games']}\n")
        f.write("\n")
        
        # Additional statistical test results summary
        if 'algorithm_comparison_test' in statistical_results:
            test = statistical_results['algorithm_comparison_test']
            significance_threshold = config.get('statistical', {}).get('significance_threshold', 0.05)
            is_significant = test['p_value'] < significance_threshold
            f.write(f"Statistical significance: {'Yes' if is_significant else 'No'} (p={test['p_value']:.4f}, α={significance_threshold})\n")
            if 'effect_size' in test:
                effect_size = test['effect_size']
                effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                f.write(f"Effect size: {effect_size:.4f} ({effect_magnitude})\n\n")
        
        f.write("Role Performance:\n")
        f.write("----------------\n")
        for i, row in performance_metrics['role_performance'].iterrows():
            f.write(f"{row['Role']}: Win Rate = {row['Win Rate']:.4f} (95% CI: [{row['CI Lower']:.4f}, {row['CI Upper']:.4f}])\n")
            f.write(f"  Win: {row['Win %']*100:.1f}%, Draw: {row['Draw %']*100:.1f}%, Loss: {row['Loss %']*100:.1f}%\n")
            f.write(f"  Games played: {row['Games']}\n")
        f.write("\n")
        
        f.write("Minimax Depth Performance:\n")
        f.write("------------------------\n")
        if len(performance_metrics['depth_performance']) > 0:
            for i, row in performance_metrics['depth_performance'].iterrows():
                f.write(f"Depth {row['Depth']}: Win Rate = {row['Win Rate']:.4f} (95% CI: [{row['CI Lower']:.4f}, {row['CI Upper']:.4f}])\n")
                f.write(f"  As Tiger: {row['As Tiger Win Rate']:.4f}, As Goat: {row['As Goat Win Rate']:.4f}\n")
                f.write(f"  Avg Move Time: {row['Avg Move Time (s)']:.2f}s\n")
                f.write(f"  Games played: {row['Games']}\n")
            f.write("\n")
            
            # Add pairwise depth comparisons summary
            if 'depth_comparison_tests' in statistical_results:
                f.write("Depth Comparison Statistical Tests:\n")
                f.write("--------------------------------\n")
                from statsmodels.stats.multitest import multipletests
                
                tests = statistical_results['depth_comparison_tests']
                if tests:
                    test_keys = list(tests.keys())
                    all_p_values = [test['p_value'] for test in tests.values()]
                    
                    # Apply correction if we have p-values
                    if all_p_values:
                        rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                        
                        for i, key in enumerate(test_keys):
                            test = tests[key]
                            d1, d2 = key.split('_vs_')
                            corrected_p = corrected_p_values[i]
                            is_significant = corrected_p < significance_threshold
                            significance = "SIGNIFICANT" if is_significant else "not significant"
                            
                            f.write(f"Depth {d1} vs Depth {d2}: p={test['p_value']:.4f} (corrected: {corrected_p:.4f}) - {significance}\n")
                            if is_significant and 'depth1_mean' in test and 'depth2_mean' in test:
                                better_depth = d1 if test['depth1_mean'] > test['depth2_mean'] else d2
                                f.write(f"  Depth {better_depth} performs better (effect size: {test['effect_size']:.4f})\n")
                        f.write("\n")
        else:
            f.write("No depth performance data available\n\n")
        
        f.write("Top MCTS Configurations:\n")
        f.write("----------------------\n")
        for i, config in enumerate(performance_metrics['top_mcts_configs']):
            f.write(f"{i+1}. {config['config_id']} - Win rate: {config['win_rate']:.4f}\n")
            f.write(f"   Parameters: ")
            if 'exploration_weight' in config:
                f.write(f"Exploration Weight = {config['exploration_weight']}, ")
            if 'rollout_policy' in config:
                f.write(f"Rollout Policy = {config['rollout_policy']}, ")
            if 'rollout_depth' in config:
                f.write(f"Rollout Depth = {config['rollout_depth']}")
            f.write("\n\n")
        
        f.write("Top Minimax Configurations:\n")
        f.write("-------------------------\n")
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
        
        f.write("Statistical Results Overview:\n")
        f.write("--------------------------\n")
        significance_threshold = config.get('statistical', {}).get('significance_threshold', 0.05)
        f.write(f"Significance level (α): {significance_threshold}\n")
        f.write("Multiple comparison correction: Benjamini-Hochberg procedure\n\n")
        
        # Add key significant findings
        f.write("Key Significant Findings:\n")
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
        
        if significant_findings:
            for finding in significant_findings:
                f.write(f"{finding}\n")
        else:
            f.write("No findings remained statistically significant after correction for multiple comparisons.\n")
        f.write("\n")
        
        f.write("For complete statistical validation details, see the statistical_validation.txt report.\n\n")
        
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