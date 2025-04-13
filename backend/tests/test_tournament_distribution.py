#!/usr/bin/env python3
"""
Test to verify that the MCTS tournament distributes games evenly across matchups.
This ensures that our time-based, evenly distributed scheduling is working properly.
"""

import os
import sys
import unittest
import pandas as pd
import json
import glob
from collections import defaultdict

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestMCTSTournamentDistribution(unittest.TestCase):
    """Test case for verifying MCTS tournament distribution."""
    
    def find_latest_tournament_file(self):
        """Find the most recent MCTS tournament CSV file."""
        # Look in both possible locations (root level or in simulation_results folder)
        pattern1 = os.path.join("simulation_results", "mcts_tournament", "*.csv")
        pattern2 = os.path.join("backend", "simulation_results", "mcts_tournament", "*.csv")
        
        # Look for files matching either pattern
        files1 = glob.glob(pattern1)
        files2 = glob.glob(pattern2)
        
        # Combine and find the most recent
        all_files = files1 + files2
        if not all_files:
            raise FileNotFoundError("No tournament files found")
        
        # Get the most recent file by modification time
        return max(all_files, key=os.path.getmtime)
    
    def test_game_distribution(self):
        """Test that games are evenly distributed across matchups."""
        try:
            # Find the most recent tournament file
            tournament_file = self.find_latest_tournament_file()
            print(f"Analyzing file: {tournament_file}")
            
            # Read the CSV file
            df = pd.read_csv(tournament_file)
            
            # Count configurations
            unique_configs = set()
            for config_str in df['tiger_config'].unique():
                unique_configs.add(config_str)
            for config_str in df['goat_config'].unique():
                unique_configs.add(config_str)
                
            print(f"Found {len(unique_configs)} unique configurations")
            
            # Analyze matchups
            matchups = defaultdict(lambda: {"total": 0, "tiger_games": 0, "goat_games": 0})
            
            for _, row in df.iterrows():
                tiger_config = row['tiger_config']
                goat_config = row['goat_config']
                
                # Create a sorted tuple to uniquely identify the matchup regardless of role
                configs = tuple(sorted([tiger_config, goat_config]))
                
                # Update counts
                matchups[configs]["total"] += 1
                
                # Track which config was tiger
                if tiger_config == configs[0]:
                    matchups[configs]["tiger_games"] += 1
                else:
                    matchups[configs]["goat_games"] += 1
            
            # Calculate statistics
            counts = [data["total"] for data in matchups.values()]
            tiger_counts = [data["tiger_games"] for data in matchups.values()]
            goat_counts = [data["goat_games"] for data in matchups.values()]
            
            # Calculate max difference between matchups
            if counts:
                min_count = min(counts) 
                max_count = max(counts)
                game_count_diff = max_count - min_count
            else:
                min_count = max_count = game_count_diff = 0
                
            # Calculate max difference between tiger and goat games for any matchup
            role_diffs = [abs(data["tiger_games"] - data["goat_games"]) for data in matchups.values()]
            max_role_diff = max(role_diffs) if role_diffs else 0
            
            # Calculate rounds (total games divided by 2)
            rounds = [data["total"] // 2 for data in matchups.values()]
            min_round = min(rounds) if rounds else 0
            max_round = max(rounds) if rounds else 0
            round_diff = max_round - min_round
            
            # Print results
            print(f"Total matchups: {len(matchups)}")
            print(f"Total games: {len(df)}")
            print(f"Games per matchup: min={min_count}, max={max_count}, diff={game_count_diff}")
            print(f"Rounds per matchup: min={min_round}, max={max_round}, diff={round_diff}")
            print(f"Max difference between tiger/goat roles: {max_role_diff}")
            
            # Extract config details for better reporting
            matchup_details = []
            for configs, data in matchups.items():
                config1 = json.loads(configs[0])
                config2 = json.loads(configs[1])
                
                # Simplify config representation - include exploration weight and guided strictness
                config1_desc = f"{config1['rollout_policy']}-{config1['iterations']}-{config1['rollout_depth']}-" \
                               f"{config1.get('exploration_weight', 1.414)}-{config1.get('guided_strictness', 0.5)}"
                config2_desc = f"{config2['rollout_policy']}-{config2['iterations']}-{config2['rollout_depth']}-" \
                               f"{config2.get('exploration_weight', 1.414)}-{config2.get('guided_strictness', 0.5)}"
                
                round_num = data["total"] // 2
                role_balance = abs(data["tiger_games"] - data["goat_games"])
                
                matchup_details.append({
                    "config1": config1_desc,
                    "config2": config2_desc,
                    "total_games": data["total"],
                    "round": round_num,
                    "config1_as_tiger": data["tiger_games"] if configs[0] == configs[0] else data["goat_games"],
                    "config2_as_tiger": data["goat_games"] if configs[0] == configs[0] else data["tiger_games"],
                    "role_balance_diff": role_balance
                })
            
            print("\nMatchup Details:")
            for detail in sorted(matchup_details, key=lambda x: x["total_games"]):
                print(f"  {detail['config1']} vs {detail['config2']}: {detail['total_games']} games (round {detail['round']}), "
                      f"tiger/goat: {detail['config1_as_tiger']}/{detail['config2_as_tiger']}, "
                      f"role diff: {detail['role_balance_diff']}")
            
            # Assert that the difference between rounds is at most 1
            # This ensures matchups progress through rounds in lockstep
            self.assertLessEqual(round_diff, 1, 
                               f"Round distribution too uneven: min={min_round}, max={max_round}, diff={round_diff}")
            
            # Assert that the total game count difference is at most 2
            # With round-based scheduling, this should never exceed 2
            self.assertLessEqual(game_count_diff, 2, 
                               f"Game distribution too uneven: min={min_count}, max={max_count}, diff={game_count_diff}")
            
            # Assert that the difference between tiger and goat roles within any matchup is at most 1
            # This ensures roles are balanced within each matchup
            self.assertLessEqual(max_role_diff, 1,
                               f"Role balance too uneven: max difference={max_role_diff}")
            
        except FileNotFoundError:
            self.skipTest("No tournament files found to analyze")

if __name__ == "__main__":
    unittest.main() 