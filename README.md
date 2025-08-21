BaghChal AI Project

For the full MSc Project report detailing the development and evaluation of AI strategies for BaghChal, please refer to this link:

https://docs.google.com/document/d/16Q7J666m4caudsTJJBtiKq06XE86Q-x9J2fuj6aGWMM/edit?tab=t.0


BaghChal is a strategically complex asymmetric Nepali board game (~10⁴¹ states) featuring distinct Tiger/Goat objectives. Despite its depth, there are currently no rigorous AI benchmarks. This project addresses that gap by developing, systematically tuning, and comparatively evaluating non-neural strategies.

Key Highlights:

Implemented Agents: Minimax (with alpha-beta pruning, phase-aware heuristics) and Monte Carlo Tree Search (MCTS-UCT).

Genetic Algorithm (GA): Used to optimize Minimax heuristic evaluation parameters, improving performance by ~10%.

MCTS Tuning: Extensive tournaments identified the optimal configuration under time constraints, including a lightweight heuristic rollout policy, shallow simulation depth (4 plies), and low exploration (Cp=1.0).

Q-learning Agent: Used as a baseline to verify the feasibility of learning in BaghChal using engineered features.

Competition Results: GA-tuned Minimax faced off against elite MCTS configurations, with MCTS showing a modest but statistically significant advantage overall. A strong Goat player advantage was consistently observed.

Conclusion:

This study establishes a reproducible framework for evaluating BaghChal AI, validates the use of GA-based parameter tuning, and provides insights into the trade-offs between classical search and MCTS for complex asymmetric games.
