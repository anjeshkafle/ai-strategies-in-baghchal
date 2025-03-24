#!/usr/bin/env python3
import sys
import os
from typing import List, Dict, Optional
import json
import time

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mcts_agent import MCTSAgent, MCTSNode
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState

#-----------------------------------------------
# CONFIGURATION SETTINGS - MODIFY THESE AS NEEDED
#-----------------------------------------------

# Board states in an easy-to-edit format
BOARD_STRING_1 = [
    "TGG_T",
    "G__TG",
    "G___G",
    "G___G",
    "TGGGT"
]

BOARD_STRING_2 = [
    "T___T",
    "__G__",
    "_____",
    "_____",
    "T___T"
]

BOARD_STRING_3 = [
    "T___T",
    "_____",
    "_____",
    "_____",
    "TG__T"
]

# Select which board to use (directly set to the board string variable)
BOARD_TO_USE = BOARD_STRING_1

# Configure game state settings
GAME_PHASE = "PLACEMENT"  # "PLACEMENT" or "MOVEMENT"
TURN = "GOAT"            # "GOAT" or "TIGER"
GOATS_PLACED = 1
GOATS_CAPTURED = 0

# Select which agent(s) to run (True/False)
RUN_MINIMAX = False
RUN_MCTS = True

# Agent parameters
# Minimax parameters
MINIMAX_MAX_DEPTH = 5

# MCTS parameters
MCTS_ITERATIONS = 1000
MCTS_EXPLORATION_WEIGHT = 1.414
MCTS_ROLLOUT_POLICY = "guided"  # "random" or "guided"
MCTS_MAX_ROLLOUT_DEPTH = 2
MCTS_GUIDED_STRICTNESS = 0
MCTS_MAX_TIME_SECONDS = 50

#-----------------------------------------------
# HELPER FUNCTIONS
#-----------------------------------------------

def string_board_to_game_state(string_board: List[str], phase=GAME_PHASE, turn=TURN, goats_placed=GOATS_PLACED, goats_captured=GOATS_CAPTURED):
    """
    Convert a string-based board representation to a GameState object.
    
    Args:
        string_board: A list of strings representing the board state.
        phase: The game phase (default: "PLACEMENT").
        turn: Whose turn it is (default: "TIGER").
        goats_placed: Number of goats placed (default: 1).
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


def print_board(state: GameState) -> None:
    """
    Print the current board state in a readable format.
    """
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


def format_move(move: Optional[Dict]) -> str:
    """
    Format a move dictionary in a human-readable way.
    """
    if move is None:
        return "No valid moves available"
    
    if move["type"] == "placement":
        return f"Place goat at ({move['x']}, {move['y']})"
    else:  # movement
        from_x, from_y = move["from"]["x"], move["from"]["y"]
        to_x, to_y = move["to"]["x"], move["to"]["y"]
        
        if move.get("capture"):
            cap_x, cap_y = move["capture"]["x"], move["capture"]["y"]
            return f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y}), capturing goat at ({cap_x}, {cap_y})"
        else:
            return f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y})"


def print_best_move_sequence(initial_state: GameState, max_depth: int) -> None:
    """
    Print the best move sequence from the current state up to max_depth.
    
    Args:
        initial_state: The starting game state
        max_depth: The maximum search depth
    """
    print("\n=== Best Move Sequence ===")
    
    current_state = initial_state.clone()
    current_depth = max_depth
    
    for i in range(max_depth):
        # Create a new agent with decreasing depth
        agent = MinimaxAgent(max_depth=current_depth)
        
        # Get the best move
        best_move = agent.get_move(current_state)
        
        if best_move is None:
            print(f"  {i+1}. No valid moves available")
            break
            
        # Print the move with player and score
        print(f"\n  {i+1}. {current_state.turn} plays: {format_move(best_move)} (score: {agent.best_score:.1f})")
        
        # Apply the move
        current_state.apply_move(best_move)
        
        # Print the board state after this move
        print(f"\nBoard after move {i+1}:")
        print_board(current_state)
        
        # Decrease depth for next iteration
        current_depth -= 1
    
    # Print the final position evaluation
    final_eval_agent = MinimaxAgent(max_depth=1)
    static_eval = final_eval_agent.evaluate(current_state)
    print(f"\nFinal position static evaluation: {static_eval:.1f}")


def print_mcts_stats(root_node: MCTSNode) -> None:
    """
    Print statistics about the MCTS tree.
    
    Args:
        root_node: The root node of the MCTS tree
    """
    print("\n=== MCTS Statistics ===")
    print(f"Total root node visits: {root_node.visits}")
    
    # Sort children by number of visits
    if root_node.children:
        sorted_children = sorted(root_node.children, key=lambda c: c.visits, reverse=True)
        
        # Print top moves
        print("\nTop moves by visit count:")
        for i, child in enumerate(sorted_children):  # Show all moves
            win_rate = child.value / child.visits if child.visits > 0 else 0
            print(f"  {i+1}. {format_move(child.move)} - Visits: {child.visits}, Win rate: {win_rate:.2f}")
    else:
        print("No children nodes found.")


def run_minimax_test(game_state):
    """Run the minimax agent test."""
    print("\n" + "="*50)
    print("RUNNING MINIMAX AGENT TEST")
    print("="*50)
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"\nValid moves for {game_state.turn}: {len(valid_moves)}")
    
    # Get capture moves
    capture_moves = [move for move in valid_moves if move.get("capture")]
    print(f"Capture moves available: {len(capture_moves)}")
    
    if capture_moves:
        print("\nCapture moves:")
        for i, move in enumerate(capture_moves):
            print(f"  {i+1}. {format_move(move)}")
    
    # Define search depth
    max_depth = MINIMAX_MAX_DEPTH
    
    print(f"\nInitializing Minimax agent with max depth: {max_depth}")
    
    # Create the minimax agent
    agent = MinimaxAgent(max_depth=max_depth)
    
    # Get the best move
    best_move = agent.get_move(game_state)
    
    # Print the best move
    print("\nBest move according to Minimax agent:")
    if best_move:
        print(format_move(best_move))
        print(f"Move score: {agent.best_score}")
        
        # Apply the move to see the result
        new_state = game_state.clone()
        new_state.apply_move(best_move)
        print("\nBoard state after applying the best move:")
        print_board(new_state)
        
        # Check if the move was a capture move
        if best_move.get("capture"):
            print("✓ The agent chose a capture move!")
        else:
            if capture_moves:
                print("✗ The agent did NOT choose a capture move, even though captures are available.")
            else:
                print("(No capture moves were available)")
                
        # Print the best move sequence
        print_best_move_sequence(game_state, max_depth)
    else:
        print("No move was returned by the agent.")


def run_mcts_test(game_state):
    """Run the MCTS agent test."""
    print("\n" + "="*50)
    print("RUNNING MCTS AGENT TEST")
    print("="*50)
    
    # Get all valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"\nValid moves for {game_state.turn}: {len(valid_moves)}")
    
    # Get capture moves
    capture_moves = [move for move in valid_moves if move.get("capture")]
    print(f"Capture moves available: {len(capture_moves)}")
    
    if capture_moves:
        print("\nCapture moves:")
        for i, move in enumerate(capture_moves):
            print(f"  {i+1}. {format_move(move)}")
    
    # Configure MCTS parameters
    iterations = MCTS_ITERATIONS
    exploration_weight = MCTS_EXPLORATION_WEIGHT
    rollout_policy = MCTS_ROLLOUT_POLICY
    max_rollout_depth = MCTS_MAX_ROLLOUT_DEPTH
    guided_strictness = MCTS_GUIDED_STRICTNESS
    
    print(f"\nInitializing MCTS agent with {iterations} iterations")
    print(f"Exploration weight: {exploration_weight}")
    print(f"Rollout policy: {rollout_policy}")
    print(f"Maximum rollout depth: {max_rollout_depth}")
    print(f"Guided strictness: {guided_strictness}")
    
    # Create the MCTS agent
    agent = MCTSAgent(
        iterations=iterations,
        exploration_weight=exploration_weight,
        rollout_policy=rollout_policy,
        max_rollout_depth=max_rollout_depth,
        guided_strictness=guided_strictness
    )
    
    # Print minimax evaluation of each move for reference
    print("\nMinimax evaluation of available moves:")
    for i, move in enumerate(valid_moves):
        minimax_score = agent.evaluate_move(game_state, move)
        capture_str = " (CAPTURE)" if move.get("capture") else ""
        print(f"  {i+1}. {format_move(move)}{capture_str} - Score: {minimax_score}")
    
    # Manually create the MCTS tree to access statistics
    root = MCTSNode(game_state)
    
    # Record start time
    start_time = time.time()
    
    # Set a max time limit
    max_time_seconds = MCTS_MAX_TIME_SECONDS
    
    # Run MCTS iterations with time limit
    iterations_completed = 0
    for i in range(iterations):
        # Check if we're approaching time limit (check every 10 iterations)
        if i % 10 == 0 and time.time() - start_time > max_time_seconds:
            print(f"Time limit reached after {i} iterations")
            break
            
        # Selection phase - select a promising leaf node
        node = root
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.select_child(exploration_weight)
        
        # Expansion phase - if node is not terminal and has untried moves
        if not node.state.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
        
        # Simulation phase - perform a rollout from the new node
        result = agent.rollout(node.state)
        
        # Backpropagation phase - update statistics up the tree
        while node is not None:
            node.update(result)
            node = node.parent
            
        iterations_completed = i + 1
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print performance information
    print(f"\nMCTS completed {iterations_completed} iterations in {elapsed_time:.2f} seconds")
    if iterations_completed > 0:
        print(f"Average time per iteration: {(elapsed_time/iterations_completed)*1000:.2f} ms")
    else:
        print("No iterations completed")
    
    # Print MCTS statistics
    print_mcts_stats(root)
    
    # Select the best move based on most visits
    if root.children:
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move
        
        # Print the best move
        print("\nBest move according to MCTS agent:")
        print(format_move(best_move))
        print(f"Visits: {best_child.visits} ({best_child.visits/root.visits:.1%} of total)")
        print(f"Win rate: {best_child.value/best_child.visits:.2f}")
        
        # Apply the move to see the result
        new_state = game_state.clone()
        new_state.apply_move(best_move)
        print("\nBoard state after applying the best move:")
        print_board(new_state)
        
        # Check if the move was a capture move
        if best_move.get("capture"):
            print("✓ The agent chose a capture move!")
        else:
            if capture_moves:
                print("✗ The agent did NOT choose a capture move, even though captures are available.")
            else:
                print("(No capture moves were available)")
    else:
        print("No move was returned by the agent.")


def main():
    """Run agent tests with configurable parameters."""
    # Create a game state from the board
    game_state = string_board_to_game_state(
        BOARD_TO_USE, 
        phase=GAME_PHASE,
        turn=TURN,
        goats_placed=GOATS_PLACED,
        goats_captured=GOATS_CAPTURED
    )
    
    # Print the board state for verification
    print("\n" + "="*50)
    print("GAME STATE INFORMATION")
    print("="*50)
    print_board(game_state)
    
    # Print state information
    print(f"Turn: {game_state.turn}")
    print(f"Phase: {game_state.phase}")
    print(f"Goats placed: {game_state.goats_placed}")
    print(f"Goats captured: {game_state.goats_captured}")
    print(f"Game status: {game_state.game_status}")
    
    # Run the selected agent tests
    if RUN_MINIMAX:
        run_minimax_test(game_state)
    
    if RUN_MCTS:
        run_mcts_test(game_state)


if __name__ == "__main__":
    main() 