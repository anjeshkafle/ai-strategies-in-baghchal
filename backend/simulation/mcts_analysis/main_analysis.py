"""
Main script for MCTS tournament analysis.
"""
import os
import pandas as pd
from . import mcts_analysis
from . import visualization

def run_analysis(tournament_file, output_dir):
    """
    Run complete analysis pipeline on tournament results.
    
    Args:
        tournament_file: Path to tournament CSV file
        output_dir: Directory for output files
    
    Returns:
        Dictionary with analysis results including top configurations
    """
    # Create output directories
    data_dir = os.path.join(output_dir, 'data')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading tournament data from {tournament_file}...")
    
    # Load and process data
    tournament_data = mcts_analysis.load_and_preprocess_data(tournament_file)
    
    print(f"Analyzing {len(tournament_data)} games between MCTS configurations...")
    
    # Calculate metrics
    win_rates = mcts_analysis.calculate_win_rates(tournament_data)
    print(f"Identified {len(win_rates)} unique configurations")
    
    statistical_results = mcts_analysis.perform_statistical_tests(tournament_data)
    print("Statistical tests completed")
    
    elo_ratings = mcts_analysis.calculate_elo_ratings(tournament_data)
    print("Elo ratings calculated")
    
    # Generate composite scores
    composite_scores = mcts_analysis.generate_composite_scores(win_rates, elo_ratings)
    print("Composite scores generated")
    
    # Select top configurations
    top_configs = mcts_analysis.select_top_configurations(composite_scores)
    print(f"Selected top {len(top_configs)} configurations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualization.create_win_rate_bar_chart(win_rates, figures_dir)
    visualization.create_parameter_performance_charts(tournament_data, statistical_results, figures_dir)
    visualization.create_elo_rating_chart(elo_ratings, figures_dir)
    visualization.create_composite_score_chart(composite_scores, top_configs, figures_dir)
    visualization.create_heatmap(tournament_data, figures_dir)
    
    # Save processed data
    print("\nSaving processed data...")
    win_rates.to_csv(os.path.join(data_dir, "win_rates.csv"), index=False)
    elo_ratings.to_csv(os.path.join(data_dir, "elo_ratings.csv"), index=False)
    composite_scores.to_csv(os.path.join(data_dir, "composite_scores.csv"), index=False)
    pd.DataFrame(top_configs).to_csv(os.path.join(data_dir, "top_configs.csv"), index=False)
    
    # Print statistical results
    print("\nStatistical Results:")
    if 'depth_ttest' in statistical_results:
        depth_ttest = statistical_results['depth_ttest']
        print(f"Rollout Depth T-Test: t={depth_ttest['statistic']:.4f}, p={depth_ttest['p_value']:.4f}")
        
    if 'policy_anova' in statistical_results:
        policy_anova = statistical_results['policy_anova']
        print(f"Rollout Policy ANOVA: F={policy_anova['statistic']:.4f}, p={policy_anova['p_value']:.4f}")
        
    if 'exploration_weight_ttests' in statistical_results:
        print("Exploration Weight T-Tests:")
        for key, result in statistical_results['exploration_weight_ttests'].items():
            print(f"  {key}: t={result['statistic']:.4f}, p={result['p_value']:.4f}")
    
    # Generate a summary report
    with open(os.path.join(output_dir, "mcts_analysis_summary.txt"), "w") as f:
        f.write("MCTS Tournament Analysis Summary\n")
        f.write("===============================\n\n")
        
        f.write(f"Tournament data: {tournament_file}\n")
        f.write(f"Games analyzed: {len(tournament_data)}\n")
        f.write(f"Unique configurations: {len(win_rates)}\n\n")
        
        f.write("Statistical Results:\n")
        f.write("-----------------\n")
        if 'depth_ttest' in statistical_results:
            depth_ttest = statistical_results['depth_ttest']
            f.write(f"Rollout Depth T-Test: t={depth_ttest['statistic']:.4f}, p={depth_ttest['p_value']:.4f}\n")
            f.write(f"  Depth 4 win rate: {depth_ttest['depth4_win_rate']:.4f}\n")
            f.write(f"  Depth 6 win rate: {depth_ttest['depth6_win_rate']:.4f}\n\n")
            
        if 'policy_anova' in statistical_results:
            policy_anova = statistical_results['policy_anova']
            f.write(f"Rollout Policy ANOVA: F={policy_anova['statistic']:.4f}, p={policy_anova['p_value']:.4f}\n")
            f.write(f"  Policies tested: {', '.join(policy_anova['policies'])}\n\n")
            
        if 'exploration_weight_ttests' in statistical_results:
            f.write("Exploration Weight T-Tests:\n")
            for key, result in statistical_results['exploration_weight_ttests'].items():
                f.write(f"  {key}: t={result['statistic']:.4f}, p={result['p_value']:.4f}\n")
            f.write("\n")
        
        f.write("Top Configurations:\n")
        f.write("-----------------\n")
        for i, config in enumerate(top_configs):
            f.write(f"{i+1}. {config['config_id']}\n")
            f.write(f"   Rollout Policy: {config['rollout_policy']}\n")
            f.write(f"   Rollout Depth: {config['rollout_depth']}\n")
            f.write(f"   Exploration Weight: {config['exploration_weight']}\n")
            f.write(f"   Composite Score: {config['composite_score']:.4f}\n")
            f.write(f"   Adjusted Win Rate (draws=0.5): {config['adjusted_win_rate']:.4f}\n")
            f.write(f"   Average Win Rate (wins only): {config['average_win_rate']:.4f}\n")
            f.write(f"   Elo Rating: {config['elo_rating']:.1f}\n\n")
        
        f.write("Analysis Outputs:\n")
        f.write("---------------\n")
        f.write(f"Data files in: {data_dir}\n")
        f.write(f"Visualizations in: {figures_dir}\n")
    
    print(f"\nAnalysis summary saved to {os.path.join(output_dir, 'mcts_analysis_summary.txt')}")
    
    return {
        'win_rates': win_rates,
        'elo_ratings': elo_ratings,
        'composite_scores': composite_scores,
        'top_configs': top_configs,
        'statistical_results': statistical_results
    } 