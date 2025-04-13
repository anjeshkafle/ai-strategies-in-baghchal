"""
Analysis tools for tournament results.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import glob
import json
from pathlib import Path
from scipy import stats

class TournamentAnalyzer:
    """Analyzes tournament results to find the best configurations."""
    
    def __init__(self, results_dir: str, min_games_per_config: int = 20):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing tournament results
            min_games_per_config: Minimum number of games a config must have played
        """
        self.results_dir = results_dir
        self.min_games = min_games_per_config
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all tournament result files into a single DataFrame."""
        pattern = str(Path(self.results_dir) / "mcts_tournament" / "mcts_tournament_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            raise ValueError(f"No tournament results found in {pattern}")
            
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
                
        if not dfs:
            raise ValueError("No valid tournament results found")
            
        return pd.concat(dfs, ignore_index=True)
        
    def get_config_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate statistics for each configuration.
        
        Args:
            df: DataFrame containing all tournament results
            
        Returns:
            List of dictionaries containing config statistics
        """
        config_stats = []
        
        for config_str in df['tiger_config'].unique():
            # Get games where this config was tiger
            tiger_games = df[df['tiger_config'] == config_str]
            tiger_wins = tiger_games[tiger_games['winner'] == 'tiger'].shape[0]
            tiger_win_rate = tiger_wins / len(tiger_games) if len(tiger_games) > 0 else 0
            
            # Get games where this config was goat
            goat_games = df[df['goat_config'] == config_str]
            goat_wins = goat_games[goat_games['winner'] == 'goat'].shape[0]
            goat_win_rate = goat_wins / len(goat_games) if len(goat_games) > 0 else 0
            
            # Calculate overall statistics
            total_games = len(tiger_games) + len(goat_games)
            total_wins = tiger_wins + goat_wins
            overall_win_rate = total_wins / total_games if total_games > 0 else 0
            
            # Calculate confidence intervals
            tiger_ci = self._calculate_confidence_interval(tiger_wins, len(tiger_games))
            goat_ci = self._calculate_confidence_interval(goat_wins, len(goat_games))
            overall_ci = self._calculate_confidence_interval(total_wins, total_games)
            
            config_stats.append({
                'config': json.loads(config_str),
                'tiger_win_rate': tiger_win_rate,
                'tiger_ci': tiger_ci,
                'goat_win_rate': goat_win_rate,
                'goat_ci': goat_ci,
                'overall_win_rate': overall_win_rate,
                'overall_ci': overall_ci,
                'total_games': total_games,
                'tiger_games': len(tiger_games),
                'goat_games': len(goat_games)
            })
            
        return config_stats
        
    def _calculate_confidence_interval(self, wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson score interval for binomial proportion.
        
        Args:
            wins: Number of wins
            total: Total number of games
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if total == 0:
            return (0, 0)
            
        p = wins / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        
        return (center - spread, center + spread)
        
    def find_best_config(self) -> Dict[str, Any]:
        """
        Find the best configuration based on tournament results.
        
        Returns:
            Dictionary containing the best configuration and its statistics
            
        Raises:
            ValueError: If no valid configurations found
        """
        # Load all results
        df = self.load_all_results()
        
        # Calculate statistics for each config
        config_stats = self.get_config_stats(df)
        
        # Filter out configs with too few games
        valid_configs = [s for s in config_stats if s['total_games'] >= self.min_games]
        
        if not valid_configs:
            raise ValueError(f"No configurations found with at least {self.min_games} games")
            
        # Sort by overall win rate
        valid_configs.sort(key=lambda x: x['overall_win_rate'], reverse=True)
        
        # Return best config with full statistics
        return {
            'config': valid_configs[0]['config'],
            'stats': valid_configs[0],
            'all_configs': valid_configs
        }
        
    def print_analysis(self):
        """Print detailed analysis of tournament results."""
        try:
            best = self.find_best_config()
            print("\nTournament Analysis:")
            print("===================")
            
            print(f"\nBest Configuration:")
            print(f"  Policy: {best['config']['rollout_policy']}")
            print(f"  Iterations: {best['config']['iterations']}")
            print(f"  Depth: {best['config']['rollout_depth']}")
            print(f"  Exploration Weight: {best['config'].get('exploration_weight', 1.414)}")
            print(f"  Guided Strictness: {best['config'].get('guided_strictness', 0.5)}")
            print(f"  Overall Win Rate: {best['stats']['overall_win_rate']:.2%}")
            print(f"  95% CI: [{best['stats']['overall_ci'][0]:.2%}, {best['stats']['overall_ci'][1]:.2%}]")
            print(f"  Total Games: {best['stats']['total_games']}")
            
            print("\nAll Configurations:")
            print("------------------")
            for i, config in enumerate(best['all_configs'], 1):
                print(f"\n{i}. {config['config']['rollout_policy']}-{config['config']['iterations']}-{config['config']['rollout_depth']}")
                print(f"   Exploration Weight: {config['config'].get('exploration_weight', 1.414)}")
                print(f"   Guided Strictness: {config['config'].get('guided_strictness', 0.5)}")
                print(f"   Overall Win Rate: {config['overall_win_rate']:.2%}")
                print(f"   95% CI: [{config['overall_ci'][0]:.2%}, {config['overall_ci'][1]:.2%}]")
                print(f"   Total Games: {config['total_games']}")
                print(f"   Tiger Win Rate: {config['tiger_win_rate']:.2%}")
                print(f"   Goat Win Rate: {config['goat_win_rate']:.2%}")
                
        except Exception as e:
            print(f"Error analyzing tournament results: {e}") 