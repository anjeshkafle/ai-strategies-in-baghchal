#!/usr/bin/env python3
"""
Win Rate Predictor Training Module

This module provides functions to:
1. Train the win rate predictor with game data
2. Query the predictor for win rates
3. Test and visualize win rate predictions

Command line usage:
python -m models.train_win_rate [options]

Options:
  --predict_only: Run only the prediction example
  --train_only: Run only the training example
  --mcts_info: Show information about integrating the win rate predictor with MCTS
  --all (default): Run both prediction and training examples
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from models.game_state import GameState
from models.win_rate_predictor import get_win_rate_predictor


def get_win_rate(state: GameState) -> float:
    """
    Get the predicted win rate for a given game state.
    
    Args:
        state: The game state to predict for
        
    Returns:
        The predicted win rate (from tiger's perspective)
    """
    predictor = get_win_rate_predictor()
    return predictor.predict_win_rate(state)


def train_from_game_sequence(state_sequence: List[GameState], 
                          outcome: float,
                          temporal_weight: bool = True,
                          dry_run: bool = False) -> Dict[str, Any]:
    """
    Train the win rate predictor from a sequence of game states and the final outcome.
    
    This function supports temporal weighting, which gives more importance to states
    closer to the game outcome. For example, if you provide a sequence of 12 states
    representing a complete game, the states closer to the end will have more influence
    on the win rate table updates than the early states.
    
    Args:
        state_sequence: List of game states encountered during play
        outcome: Final outcome (1.0 for Tiger win, 0.0 for Goat win, 0.5 for draw)
        temporal_weight: Whether to weight states closer to the outcome more heavily
        dry_run: If True, don't actually update the table, just report changes
        
    Returns:
        Dictionary with update statistics
    """
    predictor = get_win_rate_predictor()
    
    # Update the predictor with the state sequence
    update_stats = predictor.update_from_game_sequence(
        state_sequence, outcome, temporal_weight, dry_run
    )
    
    # Save the updated predictor if not in dry run mode
    if not dry_run:
        predictor.save()
    
    return update_stats


def get_predictor_stats() -> Dict[str, Any]:
    """
    Get statistics about the win rate predictor table.
    
    Returns:
        Dictionary with table statistics
    """
    predictor = get_win_rate_predictor()
    return predictor.get_stats()


def test_prediction_example() -> None:
    """
    Test predicting win rates for individual game states.
    """
    from models.game_state import GameState
    
    # Sample board states in string format
    board_samples = [
        # Early game
        {
            "board": [
                "T___T",
                "_____",
                "_____",
                "_____",
                "T___T"
            ],
            "phase": "PLACEMENT",
            "turn": "GOAT",
            "goats_placed": 0,
            "goats_captured": 0,
            "description": "Initial board position"
        },
        
        # Mid placement phase
        {
            "board": [
                "T_G_T",
                "_G___",
                "G___G",
                "_____",
                "T___T"
            ],
            "phase": "PLACEMENT",
            "turn": "TIGER",
            "goats_placed": 5,
            "goats_captured": 0,
            "description": "Mid-placement phase with 5 goats placed"
        },
        
        # Late game
        {
            "board": [
                "TGG_T",
                "G__TG",
                "G___G",
                "G___G",
                "TGGGT"
            ],
            "phase": "MOVEMENT",
            "turn": "GOAT",
            "goats_placed": 20,
            "goats_captured": 2,
            "description": "Late game with 2 goats captured"
        },
        
        # Scenario: Tigers about to win
        {
            "board": [
                "T___T",
                "____G",
                "_G___",
                "____G",
                "T___T"
            ],
            "phase": "MOVEMENT",
            "turn": "TIGER",
            "goats_placed": 20,
            "goats_captured": 4,
            "description": "Tigers close to winning with 4 goats captured"
        },
        
        # Scenario: Goats with advantage
        {
            "board": [
                "TGG_T",
                "GGGGG",
                "GG_GG",
                "GGGGG",
                "TGGGT"
            ],
            "phase": "MOVEMENT",
            "turn": "TIGER",
            "goats_placed": 20,
            "goats_captured": 0,
            "description": "Goats with strong defensive formation"
        }
    ]
    
    # Convert string boards to game states and get predictions
    print("\n=== Win Rate Prediction Examples ===")
    print(f"{'Game Scenario':<30} {'Phase':<10} {'Turn':<10} {'Goats Placed':<15} {'Goats Captured':<15} {'Win Rate':<10}")
    print("-" * 90)
    
    for board_info in board_samples:
        state = string_board_to_game_state(
            board_info["board"],
            board_info["phase"],
            board_info["turn"],
            board_info["goats_placed"],
            board_info["goats_captured"]
        )
        
        # Get win rate prediction
        win_rate = get_win_rate(state)
        
        # Print details
        print(f"{board_info['description']:<30} {state.phase:<10} {state.turn:<10} "
              f"{state.goats_placed:<15} {state.goats_captured:<15} {win_rate:.4f}")


def test_training_example() -> None:
    """
    Test training the win rate predictor with a simulated game sequence.
    """
    # Create a simulated game sequence
    print("\n=== Training Example: Simulated Game Sequence ===")
    print("Creating a sequence of states representing a game where goats win")
    
    # Game states in a logical sequence (simplified for demo)
    # In a real game, these states would be consecutive plies with proper board transitions
    game_states = []
    
    # Initial state
    initial_state = string_board_to_game_state(
        [
            "T___T",
            "_____",
            "_____",
            "_____",
            "T___T"
        ],
        "PLACEMENT",
        "GOAT",
        0,
        0
    )
    game_states.append(initial_state)
    
    # State after a few goat placements
    mid_state1 = string_board_to_game_state(
        [
            "T___T",
            "_G___",
            "__G__",
            "___G_",
            "T___T"
        ],
        "PLACEMENT",
        "TIGER",
        3,
        0
    )
    game_states.append(mid_state1)
    
    # Mid-game state
    mid_state2 = string_board_to_game_state(
        [
            "T_G_T",
            "_G_G_",
            "G___G",
            "_G_G_",
            "T_G_T"
        ],
        "PLACEMENT",
        "GOAT",
        9,
        0
    )
    game_states.append(mid_state2)
    
    # State with some captures
    mid_state3 = string_board_to_game_state(
        [
            "TGG_T",
            "G__TG",
            "G___G",
            "G___G",
            "TGGGT"
        ],
        "MOVEMENT",
        "TIGER",
        20,
        2
    )
    game_states.append(mid_state3)
    
    # State closer to end game
    late_state = string_board_to_game_state(
        [
            "TGG_T",
            "GGGGG",
            "G___G",
            "GGGGG",
            "TGGGT"
        ],
        "MOVEMENT",
        "GOAT",
        20,
        2
    )
    game_states.append(late_state)
    
    # Final state (goats win by forming a strong defensive formation)
    final_state = string_board_to_game_state(
        [
            "TGG_T",
            "GGGGG",
            "GG_GG",
            "GGGGG",
            "TGGGT"
        ],
        "MOVEMENT",
        "TIGER",
        20,
        2
    )
    game_states.append(final_state)
    
    # Print the sequence
    print(f"Game sequence created with {len(game_states)} states")
    print("Plies (steps from game start):")
    for i, state in enumerate(game_states):
        print(f"  Ply {i}: {state.phase} phase, {state.turn}'s turn, "
              f"{state.goats_placed} goats placed, {state.goats_captured} goats captured")
    
    # Simulate a goat win (outcome = 0.0)
    outcome = 0.0
    print(f"\nOutcome: {'Goat win' if outcome == 0.0 else 'Tiger win' if outcome == 1.0 else 'Draw'}")
    
    # First do a dry run to see what would change
    print("\nDry run (showing what would change without updating the table):")
    update_stats = train_from_game_sequence(game_states, outcome, dry_run=True)
    
    # Print update stats
    print(f"States that would be updated: {update_stats['states_updated']}")
    print(f"New entries that would be created: {update_stats['new_entries']}")
    print(f"Existing entries that would be updated: {update_stats['updated_entries']}")
    
    # Print detailed changes
    if update_stats.get('changes'):
        print("\nDetailed changes (showing how temporal weighting affects updates):")
        for change in update_stats['changes']:
            print(f"  Ply {change['state_index']}: "
                  f"Win rate change: {change['old_win_rate']:.4f} -> {change['new_win_rate']:.4f}, "
                  f"Samples: {change['old_samples']} -> {change['new_samples']}")
    
    # Do the actual update
    print("\nPerforming actual update:")
    update_stats = train_from_game_sequence(game_states, outcome)
    
    # Print update stats
    print(f"States updated: {update_stats['states_updated']}")
    print(f"New entries created: {update_stats['new_entries']}")
    print(f"Existing entries updated: {update_stats['updated_entries']}")
    
    # Get updated table stats
    table_stats = get_predictor_stats()
    print(f"\nUpdated table stats:")
    print(f"Total entries: {table_stats['total_entries']}")
    print(f"Total samples: {table_stats['total_samples']}")
    print(f"Average win rate: {table_stats['avg_win_rate']:.4f}")
    print(f"Average confidence: {table_stats['avg_confidence']:.4f}")
    
    # Check win rates again to see how they've changed
    print("\nWin rates after training:")
    for i, state in enumerate(game_states):
        win_rate = get_win_rate(state)
        print(f"  Ply {i}: {win_rate:.4f} (lower value means goats are favored)")


def string_board_to_game_state(string_board: List[str], phase="PLACEMENT", turn="GOAT", goats_placed=0, goats_captured=0) -> GameState:
    """
    Convert a string-based board representation to a GameState object.
    
    Args:
        string_board: A list of strings representing the board state.
        phase: The game phase (default: "PLACEMENT").
        turn: Whose turn it is (default: "GOAT").
        goats_placed: Number of goats placed (default: 0).
        goats_captured: Number of goats captured (default: 0).
        
    Returns:
        A GameState object initialized with the given board and settings.
    """
    # Create an empty board
    board = [[None for _ in range(5)] for _ in range(5)]
    
    # Fill the board based on the string representation
    for y, row in enumerate(string_board):
        for x, cell in enumerate(row):
            if cell == 'T':
                board[y][x] = {"type": "TIGER"}
            elif cell == 'G':
                board[y][x] = {"type": "GOAT"}
    
    # Create the game state
    state = GameState()
    state.board = board
    state.phase = phase
    state.turn = turn
    state.goats_placed = goats_placed
    state.goats_captured = goats_captured
    
    return state


def get_mcts_integration_example() -> str:
    """
    Get example code showing how to integrate the win rate predictor with MCTS.
    
    Returns:
        String with example code
    """
    example = """
# Example: Integrating Win Rate Predictor with MCTS 

from models.win_rate_predictor import get_win_rate_predictor
from models.game_state import GameState

class MCTSWithWinRateEvaluator:
    @staticmethod
    def evaluate_state(state: GameState) -> float:
        \"\"\"
        Evaluate a state using the win rate predictor.
        This can be used as evaluation function in MCTS.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Evaluation from Tiger's perspective (higher is better for Tiger)
        \"\"\"
        # Get the win rate predictor
        predictor = get_win_rate_predictor()
        
        # Get the predicted win rate
        win_rate = predictor.predict_win_rate(state)
        
        # Optional: Combine with heuristics if win rate confidence is low
        confidence = predictor.get_confidence(state)
        
        if confidence < 0.3:
            # If confidence is low, mix with a heuristic
            from models.minimax_agent import MinimaxAgent
            heuristic_value = MinimaxAgent.evaluate_state(state)
            
            # Normalize heuristic to [0,1] range
            normalized_heuristic = (heuristic_value + 1.0) / 2.0
            
            # Weighted combination of win rate and heuristic
            # As confidence grows, we rely more on the win rate
            return win_rate * confidence + normalized_heuristic * (1.0 - confidence)
        
        return win_rate
    
    @staticmethod
    def handle_playout_result(initial_state: GameState, 
                             state_sequence: List[GameState], 
                             outcome: float) -> None:
        \"\"\"
        Use playout results to train the win rate predictor.
        This can be called after MCTS simulations.
        
        Args:
            initial_state: Starting game state
            state_sequence: Sequence of states encountered during playout
            outcome: Final game result (1.0=Tiger win, 0.0=Goat win, 0.5=Draw)
        \"\"\"
        # Get the win rate predictor
        predictor = get_win_rate_predictor()
        
        # Update the predictor with the game sequence
        predictor.update_from_game_sequence(state_sequence, outcome)
        
        # Optionally save after some number of updates
        predictor.save()

# How to use in MCTS:
#
# 1. For node evaluation (in the UCT formula):
#    node_value = MCTSWithWinRateEvaluator.evaluate_state(node.state)
#
# 2. After simulation/playout:
#    MCTSWithWinRateEvaluator.handle_playout_result(
#        root_state, 
#        state_sequence_from_playout, 
#        outcome
#    )
"""
    return example

def show_mcts_integration_info() -> None:
    """
    Display information about integrating the win rate predictor with MCTS.
    """
    print("\n=== Integrating Win Rate Predictor with MCTS ===")
    print("\nThe win rate predictor can be used as an evaluation function in MCTS:")
    print("1. Replace the default rollout policy with win rate predictions")
    print("2. Use the predictor for node evaluation in the UCT formula")
    print("3. Train the predictor with results from MCTS playouts\n")
    
    print("Example code:")
    print(get_mcts_integration_example())


if __name__ == "__main__":
    # Parse command line arguments
    run_predict = True
    run_train = True
    run_mcts_info = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--predict_only":
            run_train = False
            run_mcts_info = False
            print("Running prediction example only")
        elif sys.argv[1] == "--train_only":
            run_predict = False
            run_mcts_info = False
            print("Running training example only")
        elif sys.argv[1] == "--mcts_info":
            run_predict = False
            run_train = False
            run_mcts_info = True
            print("Showing MCTS integration information")
        elif sys.argv[1] == "--all":
            run_predict = True
            run_train = True
            run_mcts_info = True
            print("Running all examples")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python -m models.train_win_rate [--predict_only|--train_only|--mcts_info|--all]")
    
    # Run the selected examples
    if run_predict:
        test_prediction_example()
    
    if run_train:
        test_training_example()
        
    if run_mcts_info:
        show_mcts_integration_info() 