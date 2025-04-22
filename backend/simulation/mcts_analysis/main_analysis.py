"""
Main script for MCTS tournament analysis.
"""
import os
import pandas as pd
import numpy as np
from . import mcts_analysis
from . import visualization

def run_analysis(tournament_file, output_dir, config_file=None):
    """
    Run complete analysis pipeline on tournament results.
    
    Args:
        tournament_file: Path to tournament CSV file
        output_dir: Directory for output files
        config_file: Path to configuration JSON file
    
    Returns:
        Dictionary with analysis results including top configurations
    """
    # Load configuration
    config = mcts_analysis.load_config(config_file)
    
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
    
    # Generate detailed statistical validation report
    statistical_report = mcts_analysis.generate_statistical_report(statistical_results, output_dir, config)
    print(f"Statistical validation report generated: {statistical_report}")
    
    elo_ratings = mcts_analysis.calculate_elo_ratings(
        tournament_data, 
        K=config['elo']['base_k_factor'], 
        initial_rating=config['elo']['initial_rating']
    )
    print("Elo ratings calculated")
    
    # Generate composite scores
    composite_scores = mcts_analysis.generate_composite_scores(
        win_rates, 
        elo_ratings,
        win_rate_weight=config['composite_score']['win_rate_weight'],
        elo_weight=config['composite_score']['elo_weight']
    )
    print("Composite scores generated")
    
    # Select top configurations
    top_configs = mcts_analysis.select_top_configurations(
        composite_scores, 
        n=config['top_configs']['count']
    )
    print(f"Selected top {len(top_configs)} configurations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualization.create_win_rate_bar_chart(win_rates, figures_dir, config)
    visualization.create_parameter_performance_charts(tournament_data, statistical_results, figures_dir, config)
    visualization.create_elo_rating_chart(elo_ratings, figures_dir, config)
    visualization.create_composite_score_chart(composite_scores, top_configs, figures_dir, config)
    visualization.create_heatmap(tournament_data, figures_dir)
    
    # Save processed data
    print("\nSaving processed data...")
    win_rates.to_csv(os.path.join(data_dir, "win_rates.csv"), index=False)
    elo_ratings.to_csv(os.path.join(data_dir, "elo_ratings.csv"), index=False)
    composite_scores.to_csv(os.path.join(data_dir, "composite_scores.csv"), index=False)
    pd.DataFrame(top_configs).to_csv(os.path.join(data_dir, "top_configs.csv"), index=False)
    
    # Print statistical results
    significance_threshold = config['statistical']['significance_threshold']
    print("\nStatistical Results:")
    if 'depth_ttest' in statistical_results:
        depth_ttest = statistical_results['depth_ttest']
        is_significant = depth_ttest['p_value'] < significance_threshold
        significance_marker = "* SIGNIFICANT *" if is_significant else "not significant"
        print(f"Rollout Depth T-Test: t={depth_ttest['statistic']:.4f}, p={depth_ttest['p_value']:.4f} ({significance_marker})")
        
    if 'policy_anova' in statistical_results:
        policy_anova = statistical_results['policy_anova']
        is_significant = policy_anova['p_value'] < significance_threshold
        significance_marker = "* SIGNIFICANT *" if is_significant else "not significant"
        print(f"Rollout Policy ANOVA: F={policy_anova['statistic']:.4f}, p={policy_anova['p_value']:.4f} ({significance_marker})")
        
    if 'exploration_weight_ttests' in statistical_results:
        print("Exploration Weight T-Tests:")
        for key, result in statistical_results['exploration_weight_ttests'].items():
            is_significant = result['p_value'] < significance_threshold
            significance_marker = "* SIGNIFICANT *" if is_significant else "not significant"
            print(f"  {key}: t={result['statistic']:.4f}, p={result['p_value']:.4f} ({significance_marker})")
    
    # Get corrected p-values for multiple comparisons
    from statsmodels.stats.multitest import multipletests
    
    # Generate a summary report
    with open(os.path.join(output_dir, "mcts_analysis_summary.txt"), "w") as f:
        f.write("MCTS Tournament Analysis Summary\n")
        f.write("===============================\n\n")
        
        f.write(f"Tournament data: {tournament_file}\n")
        f.write(f"Games analyzed: {len(tournament_data)}\n")
        f.write(f"Unique configurations: {len(win_rates)}\n\n")
        
        f.write("Statistical Validation Summary:\n")
        f.write("--------------------------\n")
        f.write(f"Significance level (α): {significance_threshold}\n")
        f.write("Multiple comparison correction: Benjamini-Hochberg procedure\n\n")
        
        # Add key significant findings with multiple comparison correction
        f.write("Key Findings After Multiple Comparison Correction:\n")
        significant_findings = []
        
        # Depth comparison
        if 'depth_ttest' in statistical_results:
            test = statistical_results['depth_ttest']
            if test['p_value'] < significance_threshold:
                better_depth = "4" if test['depth4_win_rate'] > test['depth6_win_rate'] else "6"
                effect_size = abs(test['depth4_win_rate'] - test['depth6_win_rate'])
                effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                significant_findings.append(f"- Rollout depth {better_depth} performs significantly better (p={test['p_value']:.4f}, effect size: {effect_size:.4f} - {effect_magnitude})")
        
        # Policy comparison
        if 'policy_anova' in statistical_results:
            test = statistical_results['policy_anova']
            if test['p_value'] < significance_threshold and 'policies' in test and 'policy_win_rates' in test:
                best_idx = np.argmax(test['policy_win_rates'])
                best_policy = test['policies'][best_idx]
                significant_findings.append(f"- Rollout policy {best_policy} performs significantly better (ANOVA p={test['p_value']:.4f})")
        
        # Exploration weight comparisons
        if 'exploration_weight_ttests' in statistical_results:
            tests = statistical_results['exploration_weight_ttests']
            all_p_values = [test['p_value'] for test in tests.values()]
            
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                test_keys = list(tests.keys())
                
                for i, key in enumerate(test_keys):
                    test = tests[key]
                    if corrected_p_values[i] < significance_threshold:
                        w1, w2 = key.split('_vs_')
                        better_weight = w1 if test.get('w1_win_rate', 0) > test.get('w2_win_rate', 0) else w2
                        effect_size = abs(test.get('w1_win_rate', 0) - test.get('w2_win_rate', 0))
                        effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                        significant_findings.append(f"- Exploration weight {better_weight} performs significantly better than weight {w2 if better_weight == w1 else w1} (corrected p={corrected_p_values[i]:.4f}, effect size: {effect_size:.4f} - {effect_magnitude})")
        
        if significant_findings:
            for finding in significant_findings:
                f.write(f"{finding}\n")
        else:
            f.write("No findings remained statistically significant after correction for multiple comparisons.\n")
        f.write("\n")
        
        f.write("Statistical Test Details:\n")
        f.write("---------------------\n")
        if 'depth_ttest' in statistical_results:
            depth_ttest = statistical_results['depth_ttest']
            is_significant = depth_ttest['p_value'] < significance_threshold
            significance = "SIGNIFICANT" if is_significant else "NOT significant"
            f.write(f"Rollout Depth T-Test: t={depth_ttest['statistic']:.4f}, p={depth_ttest['p_value']:.4f} ({significance})\n")
            f.write(f"  This means the difference in win rates between depth 4 and depth 6 is {significance.lower()}.\n")
            f.write(f"  Depth 4 win rate: {depth_ttest['depth4_win_rate']:.4f}\n")
            f.write(f"  Depth 6 win rate: {depth_ttest['depth6_win_rate']:.4f}\n")
            if 'effect_size' in depth_ttest:
                effect_size = depth_ttest['effect_size']
                effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                f.write(f"  Effect size: {effect_size:.4f} ({effect_magnitude})\n")
            f.write("\n")
            
        if 'policy_anova' in statistical_results:
            policy_anova = statistical_results['policy_anova']
            is_significant = policy_anova['p_value'] < significance_threshold
            significance = "SIGNIFICANT" if is_significant else "NOT significant"
            f.write(f"Rollout Policy ANOVA: F={policy_anova['statistic']:.4f}, p={policy_anova['p_value']:.4f} ({significance})\n")
            f.write(f"  This means the differences in win rates between different policies are {significance.lower()}.\n")
            f.write(f"  Policies tested: {', '.join(policy_anova['policies'])}\n")
            
            if 'policy_win_rates' in policy_anova:
                f.write("  Policy win rates:\n")
                for i, policy in enumerate(policy_anova['policies']):
                    win_rate = policy_anova['policy_win_rates'][i]
                    f.write(f"    {policy}: {win_rate:.4f}\n")
            f.write("\n")
            
        if 'exploration_weight_ttests' in statistical_results:
            f.write("Exploration Weight T-Tests (with Benjamini-Hochberg correction):\n")
            
            # Get p-values for correction
            tests = statistical_results['exploration_weight_ttests']
            all_p_values = [test['p_value'] for test in tests.values()]
            
            # Apply correction if we have p-values
            if all_p_values:
                rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
                test_keys = list(tests.keys())
                
                for i, key in enumerate(test_keys):
                    result = tests[key]
                    original_p = result['p_value']
                    corrected_p = corrected_p_values[i]
                    is_significant = corrected_p < significance_threshold
                    significance = "SIGNIFICANT" if is_significant else "not significant"
                    w1, w2 = key.split('_vs_')
                    
                    f.write(f"  {key}: t={result['statistic']:.4f}\n")
                    f.write(f"    Original p-value: {original_p:.4f}, Corrected p-value: {corrected_p:.4f} ({significance})\n")
                    if 'w1_win_rate' in result and 'w2_win_rate' in result:
                        f.write(f"    Weight {w1} win rate: {result['w1_win_rate']:.4f}\n")
                        f.write(f"    Weight {w2} win rate: {result['w2_win_rate']:.4f}\n")
                        if is_significant:
                            better_weight = w1 if result['w1_win_rate'] > result['w2_win_rate'] else w2
                            effect_size = abs(result['w1_win_rate'] - result['w2_win_rate'])
                            effect_magnitude = "large" if effect_size > 0.2 else "medium" if effect_size > 0.1 else "small"
                            f.write(f"    Weight {better_weight} performs significantly better (effect size: {effect_size:.4f} - {effect_magnitude})\n")
            f.write("\n")
        
        f.write("Assumption Checks:\n")
        f.write("----------------\n")
        if 'assumption_checks' in statistical_results:
            assumptions = statistical_results['assumption_checks']
            if 'sample_sizes' in assumptions:
                f.write(f"Sample sizes adequate for Central Limit Theorem: {'Yes' if assumptions.get('adequate_sample_size', False) else 'No'}\n")
            if isinstance(assumptions.get('normality', ''), list) and len(assumptions['normality']) > 0:
                normal_count = sum(1 for result in assumptions['normality'] if result.get('normal', False))
                f.write(f"Normality assumption met: {normal_count}/{len(assumptions['normality'])} groups\n")
            if isinstance(assumptions.get('homogeneity_of_variance', {}), dict):
                hov = assumptions['homogeneity_of_variance']
                f.write(f"Equal variance assumption met: {'Yes' if hov.get('equal_variance', False) else 'No'}\n")
        f.write("\n")
        
        f.write("For complete statistical validation details, see the statistical_validation.txt report.\n\n")
        
        f.write("Top Configurations:\n")
        f.write("-----------------\n")
        for i, config in enumerate(top_configs):
            f.write(f"{i+1}. {config['config_id']}\n")
            
            if 'rollout_policy' in config:
                f.write(f"   Rollout Policy: {config['rollout_policy']}\n")
            
            if 'rollout_depth' in config:
                f.write(f"   Rollout Depth: {config['rollout_depth']}\n")
            
            if 'exploration_weight' in config:
                f.write(f"   Exploration Weight: {config['exploration_weight']}\n")
            
            if 'composite_score' in config:
                f.write(f"   Composite Score: {config['composite_score']:.4f}\n")
            
            if 'adjusted_win_rate' in config:
                f.write(f"   Adjusted Win Rate (draws=0.5): {config['adjusted_win_rate']:.4f}\n")
            
            if 'average_win_rate' in config:
                f.write(f"   Average Win Rate (wins only): {config['average_win_rate']:.4f}\n")
            
            if 'elo_rating' in config:
                f.write(f"   Elo Rating: {config['elo_rating']:.1f}\n")
            
            f.write("\n")
        
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