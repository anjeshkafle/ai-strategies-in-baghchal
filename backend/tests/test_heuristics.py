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
            "goats_placed": 5,
            "goats_captured": 0,
            "desc": "Early placement, no captures"
        },
        # Board 2: Optimal tiger position, no captures
        {
            "board": [
                "_____",
                "_T_T_",
                "_____",
                "_T_T_",
                "_____"
            ],
            "phase": "PLACEMENT",
            "turn": "GOAT",
            "goats_placed": 10,
            "goats_captured": 0,
            "desc": "Optimal tiger position, no captures"
        },
        # Board 3: One capture, mid-placement
        {
            "board": [
                "T___T",
                "_GG__",
                "__G__",
                "_G___",
                "T___T"
            ],
            "phase": "PLACEMENT",
            "turn": "GOAT",
            "goats_placed": 15,
            "goats_captured": 1,
            "desc": "Mid-placement with one capture"
        },
        # Board 4: Closed spaces for tigers
        {
            "board": [
                "T_G_T",
                "GGGGG",
                "_GGG_",
                "GGGGG",
                "T_G_T"
            ],
            "phase": "MOVEMENT",
            "turn": "TIGER",
            "goats_placed": 20,
            "goats_captured": 0,
            "desc": "Many closed spaces (good for goats)"
        },
        # Board 5: Three captures, movement phase
        {
            "board": [
                "T___T",
                "_G___",
                "_____",
                "___G_",
                "T___T"
            ],
            "phase": "MOVEMENT",
            "turn": "TIGER",
            "goats_placed": 20,
            "goats_captured": 3,
            "desc": "Movement phase with three captures"
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
        simple_win_rate = mcts_agent.predict_win_rate(game_state)
        advanced_win_rate = mcts_agent.predict_win_rate(game_state)
        
        print("\nWin Rate Predictions (from Tiger's perspective):")
        print(f"Win rate: {simple_win_rate:.4f}")
        
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
        
        print(f"\nExpected captures at this stage: {expected_captures:.2f}")
        print(f"Actual captures: {game_state.goats_captured}")
        print(f"Capture deficit: {capture_deficit:.2f}")
        
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