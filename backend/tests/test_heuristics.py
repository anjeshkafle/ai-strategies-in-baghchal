#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from models.mcts_agent import MCTSAgent
from game_logic import get_all_possible_moves


def string_board_to_game_state(string_board, phase="PLACEMENT", turn="TIGER", goats_placed=1, goats_captured=0):
    """Convert a string-based board representation to a GameState object."""
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


def print_board(state):
    """Print the current board state in a readable format."""
    print("Current Board State:")
    print("-" * 11)
    for y in range(GameState.BOARD_SIZE):
        row = "|"
        for x in range(GameState.BOARD_SIZE):
            piece = state.board[y][x]
            if piece is None:
                row += " |"
            elif piece["type"] == "TIGER":
                row += "T|"
            else:  # GOAT
                row += "G|"
        print(row)
        print("-" * 11)


def test_mcts_win_rate_predictor():
    """Test the MCTS agent's advanced win rate predictor on different board configurations."""
    # Create boards with different captures and positions
    test_boards = [
        # Board 1: No captures, early placement
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
            "goats_placed": 20,
            "goats_captured": 2,
            "desc": "Expect higher win rate for tigers because goat is placed in the center"
        },
        # Board 7: Negative deficit (goat advantage)
        {
            "board": [
                "____T",
                "_T___",
                "_____",
                "_____",
                "_T__T"
            ],
            "phase": "PLACEMENT",
            "turn": "GOAT",
            "goats_placed": 20,
            "goats_captured": 2,
            "desc": "Expect lower win rate for tigers because goat is placed in the edge"
        }
    ]
    
    # Create agents
    mcts_agent = MCTSAgent()
    
    print("\n===== TESTING MCTS WIN RATE PREDICTOR =====")
    print("Comparing simple vs. advanced win rate prediction")
    print("=============================================")
    
    for board_data in test_boards:
        # Create the game state
        game_state = string_board_to_game_state(
            board_data["board"],
            phase=board_data["phase"],
            turn=board_data["turn"],
            goats_placed=board_data["goats_placed"],
            goats_captured=board_data["goats_captured"]
        )
        
        # Print board and state info
        print(f"\n***** {board_data['desc']} *****")
        print_board(game_state)
        print(f"Phase: {game_state.phase}, Turn: {game_state.turn}")
        print(f"Goats placed: {game_state.goats_placed}, Goats captured: {game_state.goats_captured}")
        
        # Display threatened nodes analysis
        threatened_nodes = game_state.get_threatened_nodes()
        print("\n=== THREATENED NODES ANALYSIS ===")
        print(f"Total threatened nodes: {len(threatened_nodes)}")
        print("Threatened positions (x, y) -> landing position (landing_x, landing_y):")
        for x, y, landing_x, landing_y in threatened_nodes:
            is_landing_empty = game_state.board[landing_y][landing_x] is None
            landing_status = "empty" if is_landing_empty else "occupied"
            print(f"  ({x}, {y}) -> ({landing_x}, {landing_y}) [Landing: {landing_status}]")
        print("="*35)
        
        # Calculate win rates with both methods
        simple_win_rate = mcts_agent.predict_win_rate_basic(game_state)
        advanced_win_rate = mcts_agent.predict_win_rate_advanced(game_state)
        
        print("\nWin Rate Predictions (from Tiger's perspective):")
        print(f"Simple win rate: {simple_win_rate:.4f}")
        print(f"Advanced win rate: {advanced_win_rate:.4f}")
        
        # Get detailed heuristic values
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        
        # Calculate key heuristics
        movable_tigers = mcts_agent.minimax_agent._count_movable_tigers(all_tiger_moves)
        closed_regions = mcts_agent.minimax_agent._count_closed_spaces(game_state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        threatened_goats = mcts_agent.minimax_agent._count_threatened_goats(all_tiger_moves, game_state.turn)
        position_score = mcts_agent.minimax_agent._calculate_tiger_positional_score(game_state)
        optimal_spacing_score = mcts_agent.minimax_agent._calculate_tiger_optimal_spacing(game_state)
        edge_score = mcts_agent.minimax_agent._calculate_goat_edge_preference(game_state)
        
        print("\nContributing Heuristics:")
        print(f"Movable Tigers: {movable_tigers}/4")
        print(f"Closed Spaces: {total_closed_spaces}")
        print(f"Threatened Goats: {threatened_goats}")
        print(f"Tiger Position Score: {position_score:.3f}")
        print(f"Optimal Spacing Score: {optimal_spacing_score:.3f}")
        print(f"Goat Edge Preference: {edge_score:.3f}")
        
        # Calculate expected captures
        if game_state.goats_placed < 15:
            expected_captures = 0
        else:
            expected_captures = (game_state.goats_placed - 15) * (2 / 5)
        
        # Calculate capture deficit
        capture_deficit = game_state.goats_captured - expected_captures
        
        # Calculate the internal components for advanced win rate
        capture_effect = mcts_agent._map_captures_to_win_rate(capture_deficit, game_state.goats_captured)
        
        # Calculate the components for effective captures
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        closed_regions = mcts_agent.minimax_agent._count_closed_spaces(game_state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        placement_progress = min(1.0, game_state.goats_placed / 20)
        closed_space_value = total_closed_spaces * (0.3 + (0.5 * placement_progress))
        threatened_value = mcts_agent.minimax_agent._count_threatened_goats(all_tiger_moves, game_state.turn)
        
        # Calculate effective captures
        effective_captures = game_state.goats_captured
        effective_captures -= closed_space_value
        effective_captures += threatened_value
        real_capture_deficit = effective_captures - expected_captures
        
        # Use the real capture deficit to calculate the effect
        correct_capture_effect = mcts_agent._map_captures_to_win_rate(real_capture_deficit, game_state.goats_captured)
        
        # Calculate dynamic influence based on different values
        capture_influence_actual = mcts_agent._calculate_dynamic_influence(game_state.goats_captured)
        capture_influence_effective = mcts_agent._calculate_dynamic_influence(effective_captures)
        
        print(f"\nAdvanced Win Rate Components:")
        print(f"Expected captures at this stage: {expected_captures:.2f}")
        print(f"Actual captures: {game_state.goats_captured}")
        print(f"Threatened value: {threatened_value:.2f}")
        print(f"Closed space value: {closed_space_value:.2f}")
        print(f"Effective captures: {effective_captures:.2f}")
        print(f"Raw deficit (actual - expected): {capture_deficit:.2f}")
        print(f"Actual deficit (effective - expected): {real_capture_deficit:.2f}")
        print(f"Capture effect (non-linear mapping): {correct_capture_effect:.4f}")
        print(f"Capture influence (actual captures): {capture_influence_actual:.2%}")
        print(f"Capture influence (effective captures): {capture_influence_effective:.2%}")
        print(f"Positional influence: {(1-capture_influence_actual):.2%}")
        
        # Calculate step-by-step progression from 0.5 to final value
        capture_component = 0.5 + correct_capture_effect * capture_influence_effective
        
        # Calculate dynamic equilibrium values based on goats_placed
        # Tiger position score equilibrium: 0.5 before 10 goats placed, decreases to 0.33 by 15 goats placed
        if game_state.goats_placed < 10:
            position_equilibrium = 0.5
        elif game_state.goats_placed < 15:
            # Linear interpolation from 0.5 to 0.33 between 10 and 15 goats placed
            position_equilibrium = 0.5 - (0.17 * (game_state.goats_placed - 10) / 5)
        else:
            position_equilibrium = 0.33
            
        # Tiger optimal spacing equilibrium: 0.5 before 10 goats placed, decreases to 0.33 by 15 goats placed
        if game_state.goats_placed < 10:
            spacing_equilibrium = 0.5
        elif game_state.goats_placed < 15:
            # Linear interpolation from 0.5 to 0.33 between 10 and 15 goats placed
            spacing_equilibrium = 0.5 - (0.17 * (game_state.goats_placed - 10) / 5)
        else:
            spacing_equilibrium = 0.33
            
        # Goat edge preference equilibrium: 1.0 before 5 goats placed, 0.8 until 12 goats placed, decreases to 0.1 by 20 goats placed
        if game_state.goats_placed < 5:
            edge_equilibrium = 1.0
        elif game_state.goats_placed < 12:
            edge_equilibrium = 0.8
        elif game_state.goats_placed <= 20:
            # Linear interpolation from 0.8 to 0.1 between 12 and 20 goats placed
            edge_equilibrium = 0.8 - (0.7 * (game_state.goats_placed - 12) / 8)
        else:
            edge_equilibrium = 0.1
        
        # Calculate individual heuristic components with dynamic equilibrium points
        position_factor = position_score - position_equilibrium
        spacing_factor = optimal_spacing_score - spacing_equilibrium
        edge_factor = edge_equilibrium - edge_score
        
        # Calculate heuristic weights
        position_weight = 1.0 / 5.5     # ~0.18 of total
        spacing_weight = 1.5 / 5.5      # ~0.27 of total
        edge_weight = 3.0 / 5.5         # ~0.55 of total
        
        # Calculate weighted components
        heuristic_influence = 1.0 - capture_influence_effective
        
        # Individual weighted contributions
        position_contribution = position_factor * position_weight * heuristic_influence
        spacing_contribution = spacing_factor * spacing_weight * heuristic_influence
        edge_contribution = edge_factor * edge_weight * heuristic_influence
        
        # Apply weighted factors
        heuristic_component = 0.5 + (
            (position_factor * position_weight) + 
            (spacing_factor * spacing_weight) + 
            (edge_factor * edge_weight)
        ) * heuristic_influence
        
        # Final win rate calculation
        final_win_rate = 0.5 + ((capture_component - 0.5) + (heuristic_component - 0.5))
        final_win_rate = max(0.01, min(0.99, final_win_rate))
        
        print("\nDetailed Value Progression (from 0.5 neutral):")
        print(f"Starting neutral value: 0.5000")
        print(f"Capture component: 0.5000 → {capture_component:.4f} (effect: {capture_component - 0.5:.4f})")
        print(f"  - Non-linear capture mapping: {correct_capture_effect:.4f}")
        print(f"  - Applied influence: {capture_influence_effective:.2%}")
        
        print(f"Heuristic component: 0.5000 → {heuristic_component:.4f} (effect: {heuristic_component - 0.5:.4f})")
        print(f"  - Position factor ({position_score:.3f} - {position_equilibrium:.3f}): {position_factor:.3f}")
        print(f"    * Weighted contribution: {position_contribution:.4f}")
        print(f"  - Spacing factor ({optimal_spacing_score:.3f} - {spacing_equilibrium:.3f}): {spacing_factor:.3f}")
        print(f"    * Weighted contribution: {spacing_contribution:.4f}")
        print(f"  - Edge factor ({edge_equilibrium:.3f} - {edge_score:.3f}): {edge_factor:.3f}")
        print(f"    * Weighted contribution: {edge_contribution:.4f}")
        
        print(f"Final win rate: 0.5000 → {final_win_rate:.4f}")
        print(f"  - Capture contribution: {capture_component - 0.5:.4f}")
        print(f"  - Heuristic contribution: {heuristic_component - 0.5:.4f}")
        print(f"  - Combined shift: {final_win_rate - 0.5:.4f}")
        
        print("---------------------------------------------")


def main():
    """Test for heuristics including the new positional score and optimal spacing."""

    board_2 = [
        "T___T",
        "_____",
        "G____",
        "_____",
        "T___T"
    ]
    
    # Board weights matrix for reference (same as in the agent)
    board_weights = [
        [6, 5, 10, 5, 6],
        [5, 11, 7, 11, 5],
        [10, 7, 5, 7, 10],
        [5, 11, 7, 11, 5],
        [6, 5, 10, 5, 6]
    ]
    
    # Print the weights matrix for reference
    print("Board Weights Matrix:")
    for row in board_weights:
        print(row)
    print()
    
    # Test all boards
    test_boards = {
        "Board 2 (1 closed space working)": board_2,
    }
    
    agent = MinimaxAgent(max_depth=3)
    
    for board_name, board in test_boards.items():
        print(f"\n===== Testing {board_name} =====")
        
        # Create game state
        game_state = string_board_to_game_state(
            board, 
            phase="MOVEMENT",
            turn="TIGER",
            goats_placed=20,
            goats_captured=0
        )
        
        # Print the board
        print_board(game_state)
        
        # Display threatened nodes analysis
        threatened_nodes = game_state.get_threatened_nodes()
        print("\n=== THREATENED NODES ANALYSIS ===")
        print(f"Total threatened nodes: {len(threatened_nodes)}")
        print("Threatened positions (x, y) -> landing position (landing_x, landing_y):")
        for x, y, landing_x, landing_y in threatened_nodes:
            is_landing_empty = game_state.board[landing_y][landing_x] is None
            landing_status = "empty" if is_landing_empty else "occupied"
            print(f"  ({x}, {y}) -> ({landing_x}, {landing_y}) [Landing: {landing_status}]")
        print("="*35)
        
        # Get all tiger moves (needed for multiple heuristics)
        all_tiger_moves = get_all_possible_moves(game_state.board, "MOVEMENT", "TIGER")
        
        # Print key heuristic information
        movable_tigers = agent._count_movable_tigers(all_tiger_moves)
        closed_regions = agent._count_closed_spaces(game_state, all_tiger_moves)
        threatened_goats = agent._count_threatened_goats(all_tiger_moves, game_state.turn)
        
        # Print detailed information about each closed region
        print("\nDetailed Closed Regions:")
        for i, region in enumerate(closed_regions):
            print(f"Region {i+1}: {region}")
        
        # Calculate tiger positional score (normalized 0-1 score)
        position_score = agent._calculate_tiger_positional_score(game_state)
        position_weight = agent.dispersion_weight  # Same weight as in evaluate function
        weighted_position = position_weight * position_score
        
        # Calculate tiger optimal spacing score (normalized 0-1 score)
        optimal_spacing_score = agent._calculate_tiger_optimal_spacing(game_state)
        optimal_spacing_weight = int(agent.dispersion_weight * 1.5)  # 50% more weight than positional score
        weighted_spacing = optimal_spacing_weight * optimal_spacing_score
        
        # Calculate goat edge preference (normalized 0-1 score)
        edge_score = agent._calculate_goat_edge_preference(game_state)
        edge_weight = agent.edge_weight  # Same weight as in evaluate function
        weighted_edge = edge_weight * edge_score
        
        print(f"\nHeuristic Results:")
        print(f"Movable Tigers: {movable_tigers}")
        print(f"Threatened Goats: {threatened_goats}")
        print(f"Closed Regions: {len(closed_regions)}")
        print(f"Total Closed Positions: {sum(len(region) for region in closed_regions)}")
        
        print(f"\nNew Heuristics:")
        print(f"1. Tiger Positional Score: {position_score:.3f} (normalized 0-1)")
        print(f"   Weighted Position Score: {weighted_position:.1f} (weight: {position_weight})")
        print(f"2. Tiger Optimal Spacing: {optimal_spacing_score:.3f} (normalized 0-1)")
        print(f"   Weighted Optimal Spacing: {weighted_spacing:.1f} (weight: {optimal_spacing_weight})")
        print(f"3. Goat Edge Preference: {edge_score:.3f} (normalized 0-1)")
        print(f"   Weighted Edge Preference: {weighted_edge:.1f} (weight: {edge_weight})")
        
        # Calculate the evaluation score
        eval_score = agent.evaluate(game_state)
        print(f"\nOverall Evaluation Score: {eval_score}")
        
        # Print tiger positions and their position values
        print("\nTiger Position Values:")
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if game_state.board[y][x] is not None and game_state.board[y][x]["type"] == "TIGER":
                    print(f"Tiger at ({x}, {y}) with position value: {board_weights[y][x]}")


if __name__ == "__main__":
    main()
    test_mcts_win_rate_predictor() 