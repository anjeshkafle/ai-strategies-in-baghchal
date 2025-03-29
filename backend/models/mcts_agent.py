from typing import List, Dict, Optional, Tuple
import random
import math
import heapq
from models.game_state import GameState
from models.minimax_agent import MinimaxAgent
from game_logic import get_all_possible_moves

class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_valid_moves()
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this state have been tried."""
        return len(self.untried_moves) == 0
    
    def select_child(self, exploration_weight: float = 1.0) -> 'MCTSNode':
        """Select a child node using the standard UCB formula."""
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
                
            # Calculate win rate
            win_rate = child.value / child.visits
            
            # Convert to parent's perspective
            if self.state.turn != child.state.turn:
                win_rate = 1.0 - win_rate
                
            # Standard UCB formula
            exploration = exploration_weight * math.sqrt(log_visits / child.visits)
            
            return win_rate + exploration
        
        return max(self.children, key=ucb_score)
    
    def expand(self) -> 'MCTSNode':
        """Add a new child node and return it."""
        move = self.untried_moves.pop()
        next_state = self.state.clone()
        next_state.apply_move(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.value += result
        
    def backpropagate(self, result: float) -> None:
        """
        Update this node and all its ancestors with the result,
        flipping the perspective only when the player changes.
        """
        self.update(result)
        
        if self.parent:
            # Only flip result when the player changes
            if self.state.turn != self.parent.state.turn:
                self.parent.backpropagate(1.0 - result)
            else:
                self.parent.backpropagate(result)


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for the Bagh Chal game.
    
    This implementation uses the standard MCTS algorithm with the following components:
    1. Selection: UCB formula to select promising nodes
    2. Expansion: Create child nodes for untried moves
    3. Simulation: Random or guided rollouts to a fixed depth followed by win rate prediction
    4. Backpropagation: Update node statistics up the tree, flipping the result between levels
    
    All win rates and values are stored from each node's perspective, where:
    - Higher values (closer to 1.0) favor the player whose turn it is at that node
    - Lower values (closer to 0.0) favor the opponent
    
    The win rate predictor returns values from the Tiger's perspective, which are
    converted to the appropriate player's perspective during rollouts and backpropagation.
    """
    
    def __init__(self, iterations: int = 1000, exploration_weight: float = 1.0, 
                 rollout_policy: str = "random", max_rollout_depth: int = 6,
                 guided_strictness: float = 0.5):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rollout_policy = rollout_policy  # "random" or "guided"
        self.max_rollout_depth = max_rollout_depth  # Maximum depth for rollouts before using evaluation
        self.guided_strictness = max(0.0, min(1.0, guided_strictness))  # Clamp to [0, 1]
        self._last_state = None  # Track the last state for normalization context
        
        # Create a minimax agent for evaluation
        # We only use the evaluation function, depth is not relevant here
        self.minimax_agent = MinimaxAgent()
        
        # Cache for position evaluations during rollouts
        self._eval_cache = {}  # Format: {state_hash: evaluation_score}
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state using MCTS."""
        try:
            import time
            start_time = time.time()
            max_time_seconds = 50  # Maximum time to spend on calculation
            
            # Print current board state win rate prediction
            current_win_rate = self.predict_win_rate(state)
            print("\n======== INITIAL STATE ========")
            print(f"Player turn: {state.turn}")
            print(f"Phase: {state.phase}")
            print(f"Goats placed: {state.goats_placed}, Goats captured: {state.goats_captured}")
            print(f"Current win rate prediction: {current_win_rate:.4f}")
            if state.turn == "TIGER":
                print(f"(Higher values favor TIGER: 1.0=Tiger win, 0.0=Goat win)")
            else:
                print(f"(Lower values favor GOAT: 1.0=Tiger win, 0.0=Goat win)")
            print("===============================\n")
            
            # Check for edge cases first
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                return None
                
            # If there's only one valid move, return it without running MCTS
            if len(valid_moves) == 1:
                print(f"Only one valid move available, returning immediately")
                
                # Show win rate after applying the move
                next_state = state.clone()
                next_state.apply_move(valid_moves[0])
                after_win_rate = self.predict_win_rate(next_state)
                
                win_rate_change = after_win_rate - current_win_rate
                change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
                print(f"Win rate after move: {after_win_rate:.4f} ({change_str})")
                
                return valid_moves[0]
            
            # Create root node
            root = MCTSNode(state)
            
            # Run MCTS for specified number of iterations or until timeout
            iterations_completed = 0
            for i in range(self.iterations):
                # Check for timeout
                if i % 100 == 0:  # Check time periodically to avoid performance impact
                    current_time = time.time()
                    if current_time - start_time > max_time_seconds:
                        print(f"MCTS timeout after {i} iterations ({current_time - start_time:.2f} seconds)")
                        break
                
                # Selection phase - select a promising leaf node
                node = root
                while not node.state.is_terminal() and node.is_fully_expanded():
                    node = node.select_child(self.exploration_weight)
                
                # Expansion phase - if node is not terminal and has untried moves
                if not node.state.is_terminal() and not node.is_fully_expanded():
                    node = node.expand()
                
                # Simulation phase - perform a rollout from the new node
                result = self.rollout(node.state)
                
                # Backpropagation phase - update statistics up the tree
                node.backpropagate(result)
                    
                iterations_completed = i + 1
            
            print(f"MCTS completed {iterations_completed} iterations in {time.time() - start_time:.2f} seconds")
            
            # Debug flag - set to True to print detailed statistics
            debug = True
            
            if debug:
                print("\n========== MOVE ANALYSIS ==========")
                # Sort children by visits for display
                sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
                for i, child in enumerate(sorted_children):  # Show all moves
                    capture_str = " (CAPTURE)" if child.move.get("capture") else ""
                    # Calculate win rate properly considering perspective
                    raw_win_rate = child.value / child.visits if child.visits > 0 else 0
                    # Convert to root's perspective if players are different
                    if root.state.turn != child.state.turn:
                        mcts_win_rate = 1.0 - raw_win_rate
                    else:
                        mcts_win_rate = raw_win_rate
                    
                    # Get win rate prediction for this move
                    next_state = state.clone()
                    next_state.apply_move(child.move)
                    predicted_win_rate = self.predict_win_rate(next_state)
                    
                    # Calculate win rate change
                    win_rate_change = predicted_win_rate - current_win_rate
                    change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
                    
                    print(f"{i+1}. Move: {child.move}{capture_str}")
                    print(f"   Visits: {child.visits} ({child.visits/root.visits*100:.1f}% of total)")
                    print(f"   MCTS win rate: {mcts_win_rate:.4f}")
                    print(f"   Predicted win rate: {predicted_win_rate:.4f} ({change_str})")
                    
                    if i < 4:  # Add separator between moves, except after the last one
                        print("   ---")
                print("==================================\n")
            
            # Select the best child according to visits (standard MCTS approach)
            if not root.children:
                return None  # No valid moves
                
            # Log all capture moves for analysis
            capture_moves = [(child, child.value / child.visits if child.visits > 0 else 0) 
                            for child in root.children if child.move.get("capture")]
            if capture_moves:
                print("\n========== CAPTURE MOVES ==========")
                for child, raw_win_rate in capture_moves:
                    # Convert to root's perspective if players are different
                    if root.state.turn != child.state.turn:
                        mcts_win_rate = 1.0 - raw_win_rate
                    else:
                        mcts_win_rate = raw_win_rate
                        
                    next_state = state.clone()
                    next_state.apply_move(child.move)
                    predicted_win_rate = self.predict_win_rate(next_state)
                    
                    # Calculate win rate change
                    win_rate_change = predicted_win_rate - current_win_rate
                    change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
                    
                    print(f"  Capture move: {child.move}")
                    print(f"    Visits: {child.visits} ({child.visits/root.visits*100:.1f}% of total)")
                    print(f"    MCTS win rate: {mcts_win_rate:.4f}")
                    print(f"    Predicted win rate: {predicted_win_rate:.4f} ({change_str})")
                print("==================================\n")
            
            # Standard MCTS selection: choose child with most visits
            best_child = max(root.children, key=lambda c: c.visits)
            
            # Output details about the selection
            # Calculate win rate properly considering perspective
            raw_win_rate = best_child.value / best_child.visits if best_child.visits > 0 else 0
            # Convert to root's perspective if players are different
            if root.state.turn != best_child.state.turn:
                mcts_win_rate = 1.0 - raw_win_rate
            else:
                mcts_win_rate = raw_win_rate
                
            capture_text = "CAPTURE MOVE" if best_child.move.get("capture") else "regular move"
            
            # Show the predicted win rate for the selected move
            next_state = state.clone()
            next_state.apply_move(best_child.move)
            predicted_win_rate = self.predict_win_rate(next_state)
            
            # Calculate win rate change
            win_rate_change = predicted_win_rate - current_win_rate
            change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
            
            print("\n========== SELECTED MOVE ==========")
            print(f"Selected {capture_text} with {best_child.visits} visits ({best_child.visits/root.visits*100:.1f}% of total)")
            print(f"Move details: {best_child.move}")
            print(f"MCTS win rate: {mcts_win_rate:.4f}")
            print(f"Predicted win rate: {predicted_win_rate:.4f} ({change_str})")
            print("==================================\n")
            
            return best_child.move
            
        except Exception as e:
            # Log the error and fall back to a random move as a last resort
            print(f"MCTS error: {str(e)}. Falling back to random move.")
            # If we encounter an error, fall back to a random move
            valid_moves = state.get_valid_moves()
            if valid_moves:
                return random.choice(valid_moves)
            return None
    
    def rollout(self, state: GameState) -> float:
        """Perform a rollout from the given state based on the selected policy."""
        # Store the initial state for normalization context
        self._last_state = state.clone()
        
        if self.rollout_policy == "random":
            return self._random_rollout(state)
        elif self.rollout_policy == "guided":
            return self._guided_rollout(state)
        else:
            return self._random_rollout(state)  # Default to random
    
    def _random_rollout(self, state: GameState) -> float:
        """
        Simple random rollout with configurable depth.
        Returns win rate from starting player's perspective.
        """
        starting_player = state.turn
        current_state = state.clone()
        depth = 0
        max_depth = self.max_rollout_depth
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Check for threefold repetition in movement phase
            if current_state.phase == "MOVEMENT":
                state_hash = self._get_state_hash(current_state)
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    return 0.5  # Draw due to threefold repetition
            
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            move = random.choice(valid_moves)
            current_state.apply_move(move)
            depth += 1
        
        # Use win rate predictor at leaf node
        tiger_win_rate = self.predict_win_rate(current_state)
        
        # Convert to starting player's perspective
        if starting_player == "TIGER":
            return tiger_win_rate
        else:
            return 1.0 - tiger_win_rate
    
    def _guided_rollout(self, state: GameState) -> float:
        """
        Optimized guided rollout that uses a blend of minimax and randomness.
        The guided_strictness parameter controls how often we follow the best move vs random selection.
        Uses heap selection instead of full sorting for better performance.
        Returns win rate from starting player's perspective.
        """
        starting_player = state.turn
        current_state = state.clone()
        depth = 0
        max_depth = self.max_rollout_depth
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Check for threefold repetition in movement phase
            if current_state.phase == "MOVEMENT":
                state_hash = self._get_state_hash(current_state)
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    return 0.5  # Draw due to threefold repetition
            
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Decide whether to use guided selection or random selection
            if random.random() < self.guided_strictness and len(valid_moves) > 1:
                # Pre-allocate list for better performance
                move_count = len(valid_moves)
                move_scores = [(None, 0)] * move_count
                
                # Guided selection - evaluate all moves using cache when possible
                for i, move in enumerate(valid_moves):
                    next_state = current_state.clone()
                    next_state.apply_move(move)
                    
                    # Check cache for this position
                    state_hash = self._get_state_hash(next_state)
                    if state_hash in self._eval_cache:
                        score = self._eval_cache[state_hash]
                    else:
                        score = self.minimax_agent.evaluate(next_state)
                        # Store in cache (limit cache size to prevent memory issues)
                        if len(self._eval_cache) < 10000:  # Arbitrary limit
                            self._eval_cache[state_hash] = score
                    
                    move_scores[i] = (move, score)
                
                # Calculate the number of top moves to consider
                top_count = max(1, int(move_count * 0.3))
                
                # Use heap operations to find only the top N moves
                if current_state.turn == "TIGER":
                    # For Tiger, higher scores are better
                    top_moves_with_scores = heapq.nlargest(top_count, move_scores, key=lambda x: x[1])
                    top_moves = [move for move, _ in top_moves_with_scores]
                else:
                    # For Goat, lower scores are better
                    top_moves_with_scores = heapq.nsmallest(top_count, move_scores, key=lambda x: x[1])
                    top_moves = [move for move, _ in top_moves_with_scores]
                
                # Select randomly from top moves
                selected_move = random.choice(top_moves)
            else:
                # Random selection
                selected_move = random.choice(valid_moves)
            
            current_state.apply_move(selected_move)
            depth += 1
        
        # Use win rate predictor at leaf node
        tiger_win_rate = self.predict_win_rate(current_state)
        
        # Convert to starting player's perspective
        if starting_player == "TIGER":
            return tiger_win_rate
        else:
            return 1.0 - tiger_win_rate

    def evaluate_move(self, state: GameState, move) -> float:
        """Evaluate a move using the minimax evaluation function."""
        next_state = state.clone()
        next_state.apply_move(move)
        return self.minimax_agent.evaluate(next_state) 

    def _get_state_hash(self, state: GameState) -> str:
        """Create a hash representation of the game state for repetition detection."""
        # Only consider board state and turn during movement phase
        if state.phase == "MOVEMENT":
            # Convert board to a string representation
            board_str = ""
            for row in state.board:
                for cell in row:
                    if cell is None:
                        board_str += "_"
                    elif cell["type"] == "TIGER":
                        board_str += "T"
                    else:
                        board_str += "G"
            
            # Include turn in the hash
            return f"{board_str}_{state.turn}"
        else:
            # During placement phase, include goats_placed to ensure uniqueness
            return f"PLACEMENT_{state.goats_placed}_{state.turn}" 

    def predict_win_rate(self, state: GameState) -> float:
        """
        Simplified win rate predictor that ONLY cares about captures.
        Returns values from Tiger's perspective:
        - 0.1: No captures (best for Goat)
        - 0.8: One capture (significant disadvantage for Goat)
        - 0.85: Two captures (severe disadvantage for Goat)
        - 0.9: Three captures (near-loss for Goat)
        - 0.99: Four or more captures (Tiger victory approaching)
        """
        goats_captured = state.goats_captured
        
        # Extremely simple logic based only on captures
        if goats_captured == 0:
            return 0.1
        elif goats_captured == 1:
            return 0.8
        elif goats_captured == 2:
            return 0.85
        elif goats_captured == 3:
            return 0.9
        else:
            return 0.99 