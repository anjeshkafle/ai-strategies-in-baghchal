# Print top configurations
print("\nTop MCTS Configurations:")
for i, config in enumerate(top_configs):
    print(f"{i+1}. {config['config_id']}")
    print(f"   Rollout Policy: {config['rollout_policy']}")
    print(f"   Rollout Depth: {config['rollout_depth']}")
    print(f"   Exploration Weight: {config['exploration_weight']}")
    print(f"   Composite Score: {config['composite_score']:.4f}")
    print(f"   Adjusted Win Rate (draws=0.5): {config['adjusted_win_rate']:.4f}")
    print(f"   Average Win Rate (wins only): {config['average_win_rate']:.4f}")
    print(f"   Elo Rating: {config['elo_rating']:.1f}")
    print("") 