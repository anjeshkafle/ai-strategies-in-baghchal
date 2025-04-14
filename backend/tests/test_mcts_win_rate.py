"""
Tests for the MCTS advanced win rate predictor.

This module tests the advanced win rate predictor for the MCTS agent,
which uses sigmoid functions for smooth transitions between game stages.
"""

import math
import sys
from typing import List, Dict
from models.game_state import GameState
from models.mcts_agent import MCTSAgent
from game_logic import get_all_possible_moves

def string_board_to_game_state(string_board, phase="PLACEMENT", turn="TIGER", goats_placed=1, goats_captured=0):
    """Convert a string representation of a board to a GameState object."""
    board = []
    for y, row in enumerate(string_board):
        board_row = []
        for x, cell in enumerate(row):
            if cell == 'T':
                board_row.append({"type": "TIGER"})
            elif cell == 'G':
                board_row.append({"type": "GOAT"})
            else:
                board_row.append(None)
        board.append(board_row)
    
    game_state = GameState()
    game_state.board = board
    game_state.phase = phase
    game_state.turn = turn
    game_state.goats_placed = goats_placed
    game_state.goats_captured = goats_captured
    
    return game_state

def print_board(state):
    """Print a simple text representation of the board."""
    print("-" * 11)
    for y in range(len(state.board)):
        row = "|"
        for x in range(len(state.board[y])):
            cell = state.board[y][x]
            if cell is None:
                row += "_|"
            elif cell["type"] == "TIGER":
                row += "T|"
            else:  # GOAT
                row += "G|"
        print(row)
        print("-" * 11)

def test_mcts_win_rate_predictor():
    """
    Comprehensive test of the MCTS advanced win rate predictor.
    Tests captures, positional factors, and game progression all in one function.
    """
    # Create test scenarios representing different game situations and stages
    test_scenarios = [
        # Game progression scenarios
        {
            "name": "Starting position (0 goats)",
            "board": [
                "T___T",
                "____G",
                "____G",
                "_____",
                "T___T"
            ],
            "phase": "PLACEMENT",
            "turn": "GOAT",
            "goats_placed": 0,
            "goats_captured": 1,
            "description": "Initial board with 0 goats placed"
        }
    ]
    
    # Create MCTS agent
    mcts_agent = MCTSAgent()
    
    print("\n===== COMPREHENSIVE MCTS WIN RATE PREDICTOR TEST =====")
    print("This test evaluates the win rate predictor with smooth sigmoid transitions")
    
    # Track win rates for comparison
    all_win_rates = []
    all_game_states = []
    
    for i, scenario in enumerate(test_scenarios):
        # Create game state from scenario
        game_state = string_board_to_game_state(
            scenario["board"],
            phase=scenario["phase"],
            turn=scenario["turn"],
            goats_placed=scenario["goats_placed"],
            goats_captured=scenario["goats_captured"]
        )
        all_game_states.append(game_state)
        
        # Print scenario info and board
        print(f"\n\n{'='*80}")
        print(f"SCENARIO {i+1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*80}")
        print_board(game_state)
        print(f"Phase: {game_state.phase}, Turn: {game_state.turn}")
        print(f"Goats placed: {game_state.goats_placed}, Goats captured: {game_state.goats_captured}")
        
        # Get win rate prediction from the advanced predictor
        win_rate = mcts_agent.predict_win_rate_advanced(game_state)
        all_win_rates.append(win_rate)
        
        print(f"\n----- CAPTURE METRICS -----")
        
        # Calculate expected captures using the sigmoid function
        if game_state.goats_placed < 15:
            # Sigmoid function that stays close to 0 until approaching 15 goats
            transition_steepness = 0.5
            expected_captures = 2.0 / (1 + math.exp(-transition_steepness * (game_state.goats_placed - 15)))
        else:
            # After 15 goats, linear progression (2/5 of additional goats)
            expected_captures = (game_state.goats_placed - 15) * (2 / 5)
        
        # Calculate key heuristics
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        
        # Get threatened value
        threatened_value = mcts_agent.minimax_agent._count_threatened_goats(all_tiger_moves, game_state.turn)
        
        # Get closed spaces data
        closed_regions = mcts_agent.minimax_agent._count_closed_spaces(game_state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Calculate closed space value with sigmoid scaling
        min_closed_space_factor = 0.3
        max_closed_space_factor = 0.8
        factor_range = max_closed_space_factor - min_closed_space_factor
        
        # Sigmoid function for closed space scaling
        placement_sigmoid = 1.0 / (1 + math.exp(-0.4 * (game_state.goats_placed - 10)))
        closed_space_factor = min_closed_space_factor + (factor_range * placement_sigmoid)
        closed_space_value = total_closed_spaces * closed_space_factor
        
        # Calculate effective captures
        effective_captures = game_state.goats_captured - closed_space_value + threatened_value
        
        # Calculate capture deficit
        capture_deficit = effective_captures - expected_captures
        
        # Calculate capture influence
        capture_influence = mcts_agent._calculate_dynamic_influence(effective_captures, game_state.goats_placed)
        
        # Map capture deficit to win rate effect
        capture_effect = mcts_agent._map_captures_to_win_rate(capture_deficit, game_state.goats_captured)
        
        # Print capture metrics
        print(f"Expected captures: {expected_captures:.2f}")
        print(f"Actual captures: {game_state.goats_captured}")
        print(f"Threatened value: {threatened_value:.2f}")
        print(f"Closed spaces: {total_closed_spaces} (factor: {closed_space_factor:.2f}, value: {closed_space_value:.2f})")
        print(f"Effective captures: {effective_captures:.2f}")
        print(f"Capture deficit: {capture_deficit:.2f}")
        print(f"Capture effect (map to win rate): {capture_effect:.4f}")
        print(f"Capture influence: {capture_influence:.2%}")
        
        print(f"\n----- POSITIONAL METRICS -----")
        
        # Get positional scores
        position_score = mcts_agent.minimax_agent._calculate_tiger_positional_score(game_state)
        spacing_score = mcts_agent.minimax_agent._calculate_tiger_optimal_spacing(game_state)
        edge_score = mcts_agent.minimax_agent._calculate_goat_edge_preference(game_state)
        
        # Calculate equilibrium points using sigmoid functions
        goats_placed = game_state.goats_placed
        
        # Position equilibrium - sigmoid transition from 0.5 to 0.33
        position_max = 0.5
        position_min = 0.33
        position_range = position_max - position_min
        position_transition_steepness = 0.4
        position_transition_midpoint = 12.5
        position_transition = 1.0 / (1 + math.exp(-position_transition_steepness * (goats_placed - position_transition_midpoint)))
        position_equilibrium = position_max - (position_range * position_transition)
        
        # Same for spacing
        spacing_equilibrium = position_equilibrium
        
        # Edge equilibrium - dual sigmoid transition
        early_max = 1.0
        early_min = 0.8
        early_range = early_max - early_min
        early_steepness = 0.8
        early_midpoint = 3.0
        
        late_max = 0.8
        late_min = 0.1
        late_range = late_max - late_min
        late_steepness = 0.3
        late_midpoint = 16.0
        
        early_transition = 1.0 / (1 + math.exp(-early_steepness * (goats_placed - early_midpoint)))
        late_transition = 1.0 / (1 + math.exp(-late_steepness * (goats_placed - late_midpoint)))
        
        edge_equilibrium = early_max - (early_range * early_transition)
        edge_equilibrium = edge_equilibrium - (late_range * late_transition * early_transition)
        
        # Calculate position factors
        position_factor = position_score - position_equilibrium
        spacing_factor = spacing_score - spacing_equilibrium
        edge_factor = edge_equilibrium - edge_score
        
        # Calculate heuristic influence
        heuristic_influence = 1.0 - capture_influence
        
        # Print positional metrics
        print(f"Tiger Position Score: {position_score:.3f} (equilibrium: {position_equilibrium:.3f}, factor: {position_factor:.3f})")
        print(f"Tiger Spacing Score: {spacing_score:.3f} (equilibrium: {spacing_equilibrium:.3f}, factor: {spacing_factor:.3f})")
        print(f"Goat Edge Score: {edge_score:.3f} (equilibrium: {edge_equilibrium:.3f}, factor: {edge_factor:.3f})")
        print(f"Positional influence: {heuristic_influence:.2%}")
        
        # Calculate components for the final win rate
        capture_component = 0.5 + capture_effect * capture_influence
        
        # Base weight proportions for heuristics
        base_position_weight = 1.0 / 5.5
        base_spacing_weight = 1.5 / 5.5
        base_edge_weight = 3.0 / 5.5
        
        # Dynamic weight factors
        early_game_ratio = max(0.0, 1.0 - (goats_placed / 15))
        late_game_ratio = min(1, goats_placed / 15)
        
        position_weight_factor = 1.0 + (0.5 * early_game_ratio)
        edge_weight_factor = 1.0 + (1.0 * early_game_ratio)
        spacing_weight_factor = 1.0 + (0.7 * late_game_ratio)
        
        # Apply dynamic factors
        position_weight = base_position_weight * position_weight_factor
        spacing_weight = base_spacing_weight * spacing_weight_factor
        edge_weight = base_edge_weight * edge_weight_factor
        
        # Extra boost for early placement
        early_placement_boost = max(0.0, 3.0 * (1.0 - min(1.0, goats_placed / 5)))
        edge_weight += base_edge_weight * early_placement_boost
        
        # Calculate weighted contributions
        position_contribution = position_factor * position_weight * heuristic_influence
        spacing_contribution = spacing_factor * spacing_weight * heuristic_influence
        edge_contribution = edge_factor * edge_weight * heuristic_influence
        
        heuristic_sum = position_contribution + spacing_contribution + edge_contribution
        heuristic_component = 0.5 + heuristic_sum
        
        # Final win rate calculation (manually calculated for comparison)
        calculated_win_rate = 0.5 + ((capture_component - 0.5) + (heuristic_component - 0.5))
        calculated_win_rate = max(0.01, min(0.99, calculated_win_rate))
        
        print(f"\n----- FINAL WIN RATE CALCULATION -----")
        print(f"Capture component:      0.5 + {capture_effect:.4f} * {capture_influence:.2f} = {capture_component:.4f}")
        print(f"Position contribution:  {position_factor:.4f} * {position_weight:.4f} * {heuristic_influence:.2f} = {position_contribution:.4f}")
        print(f"Spacing contribution:   {spacing_factor:.4f} * {spacing_weight:.4f} * {heuristic_influence:.2f} = {spacing_contribution:.4f}")
        print(f"Edge contribution:      {edge_factor:.4f} * {edge_weight:.4f} * {heuristic_influence:.2f} = {edge_contribution:.4f}")
        print(f"Heuristic component:    0.5 + {heuristic_sum:.4f} = {heuristic_component:.4f}")
        print(f"Calculated win rate:    0.5 + ({capture_component - 0.5:.4f} + {heuristic_component - 0.5:.4f}) = {calculated_win_rate:.4f}")
        print(f"Actual win rate:        {win_rate:.4f}")
        
    # Compare win rates between scenarios
    if len(all_win_rates) > 1:
        print("\n\n----- WIN RATE PROGRESSION & COMPARISONS -----")
        print(f"{'Scenario':<30} | {'Win Rate':<10} | {'Description'}")
        print("-" * 80)
        for i, (win_rate, scenario) in enumerate(zip(all_win_rates, test_scenarios)):
            print(f"{i+1}. {scenario['name']:<25} | {win_rate:.4f}    | {scenario['description']}")
        
        # Calculate differences between consecutive scenarios
        print("\nWin Rate Changes Between Scenarios:")
        for i in range(1, len(all_win_rates)):
            win_rate_change = all_win_rates[i] - all_win_rates[i-1]
            print(f"{test_scenarios[i-1]['name']} → {test_scenarios[i]['name']}: {win_rate_change:+.4f}")
            
if __name__ == "__main__":
    print("===== MCTS ADVANCED WIN RATE PREDICTOR TEST =====")
    test_mcts_win_rate_predictor() 