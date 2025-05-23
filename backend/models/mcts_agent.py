from typing import List, Dict, Optional, Tuple
import random
import math
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
        # We need at least one child to select from
        if not self.children:
            return self  # Return self to avoid max() on empty sequence
        
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
        # Pop a move from untried_moves
        move = self.untried_moves.pop()
        
        # Apply the move to a new state
        next_state = self.state.clone()
        next_state.apply_move(move)
        
        # Create a new child node
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
    
    def __init__(self, iterations: int = None, exploration_weight: float = 1.0, 
                 rollout_policy: str = "random", max_rollout_depth: int = 6,
                 guided_strictness: float = 0.8, max_time_seconds: int = None):
        self.iterations = iterations if iterations is not None else float('inf')  # Default to unlimited iterations
        self.exploration_weight = exploration_weight
        self.rollout_policy = rollout_policy  # "random", "guided", or "lightweight"
        self.max_rollout_depth = max_rollout_depth  # Maximum depth for rollouts before using evaluation
        self.guided_strictness = max(0.0, min(1.0, guided_strictness))  # Clamp to [0, 1]
        self.max_time_seconds = max_time_seconds if max_time_seconds is not None else float('inf')  # Default to unlimited time
        self._last_state = None  # Track the last state for normalization context
        
        # Create a minimax agent for evaluation
        # We only use the evaluation function, depth is not relevant here
        self.minimax_agent = MinimaxAgent()
        
        # Cache for position evaluations during rollouts
        self._eval_cache = {}  # Format: {state_hash: evaluation_score}
        
        # Store previous search tree root for reuse
        self.previous_root = None
        
        # Ensure at least one limiting factor is set
        if self.iterations == float('inf') and self.max_time_seconds == float('inf'):
            # If neither are set, default to a reasonable time limit
            self.max_time_seconds = 10
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state using MCTS."""
        # Debug configuration - set specific types of debug information to show
        debug_config = {
            'initial_state': False,   # Show information about the initial state
            'move_analysis': False,    # Show detailed analysis of possible moves
            'selected_move': False,    # Show information about the final selected move
            'timing': False,           # Show timing and iteration information
            'tree_reuse': False,       # Show tree reuse information
            'single_move': False,      # Show information when only one move is available
            'timeout': False,          # Show timeout information
        }
        
        def debug_print_move_stats(move, win_rate, predicted_rate, change_str, visits=None, total_visits=None, is_capture=False):
            """Helper to consistently print move statistics when debug is enabled"""
            capture_str = " (CAPTURE)" if is_capture else ""
            print(f"Move: {move}{capture_str}")
            if visits is not None and total_visits is not None:
                print(f"Visits: {visits} ({visits/total_visits*100:.1f}% of total)")
            print(f"MCTS win rate: {win_rate:.4f}")
            print(f"Predicted win rate: {predicted_rate:.4f} ({change_str})")
            
        try:
            import time
            start_time = time.time()
            
            # Print current board state win rate prediction
            current_win_rate = self.predict_win_rate(state)
            if debug_config['initial_state']:
                print("\n======== INITIAL STATE ========")
                print(f"Player turn: {state.turn}")
                print(f"Phase: {state.phase}")
                print(f"Goats placed: {state.goats_placed}, Goats captured: {state.goats_captured}")
                print(f"Current win rate prediction: {current_win_rate:.4f}")
                print(f"(Higher values favor TIGER: 1.0=Tiger win, 0.0=Goat win)")
                print("===============================\n")
            
            # Check for edge cases first
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                return None
                
            # If there's only one valid move, return it without running MCTS
            if len(valid_moves) == 1:
                if debug_config['single_move']:
                    print(f"Only one valid move available, returning immediately")
                
                # Show win rate after applying the move
                next_state = state.clone()
                next_state.apply_move(valid_moves[0])
                after_win_rate = self.predict_win_rate(next_state)
                
                win_rate_change = after_win_rate - current_win_rate
                change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
                
                if debug_config['single_move']:
                    print(f"Win rate after move: {after_win_rate:.4f} ({change_str})")
                
                return valid_moves[0]
            
            # Tree reuse - attempt to find a matching child node from previous search
            root = None
            tree_reused = False
            
            if self.previous_root is not None:
                for child in self.previous_root.children:
                    # Match the board state and game parameters
                    if (self._board_matches(child.state.board, state.board) and
                        child.state.turn == state.turn and
                        child.state.phase == state.phase and
                        child.state.goats_placed == state.goats_placed and
                        child.state.goats_captured == state.goats_captured):
                        
                        # Found matching state - reuse this subtree
                        root = child
                        root.parent = None  # Detach from old tree
                        tree_reused = True
                        if debug_config['tree_reuse']:
                            print(f"Reusing subtree from previous search with {root.visits} prior visits!")
                        break
            
            # If no matching subtree found, create a new root
            if root is None:
                root = MCTSNode(state)
                
            # Run MCTS for specified number of iterations or until timeout
            max_iterations = self.iterations
            
            # Main MCTS loop - run until either iteration limit or time limit is reached
            i = 0
            while i < max_iterations:
                # Check for timeout - only check periodically to avoid performance impact
                if i % 1000 == 0:  # Check time every 1000 iterations
                    current_time = time.time()
                    if current_time - start_time > self.max_time_seconds:
                        if debug_config['timeout']:
                            print(f"MCTS timeout after {i} iterations ({current_time - start_time:.2f} seconds)")
                        break
                
                # Selection phase - select a promising leaf node
                node = root
                while not node.state.is_terminal() and node.is_fully_expanded():
                    selected_node = node.select_child(self.exploration_weight)
                    if selected_node == node:  # No children case
                        break
                    node = selected_node
                
                # Expansion phase - if node is not terminal and has untried moves
                if not node.state.is_terminal() and not node.is_fully_expanded():
                    node = node.expand()
                
                # Simulation phase - perform a rollout from the new node
                result = self.rollout(node.state)
                
                # Backpropagation phase - update statistics up the tree
                node.backpropagate(result)
                    
                i += 1
            
            if debug_config['timing']:
                print(f"MCTS completed {i} iterations in {time.time() - start_time:.2f} seconds")
                
            if debug_config['move_analysis']:
                print("\n========== MOVE ANALYSIS ==========")
                # Sort children by visits for display
                sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
                # Show top 5 moves or all moves if less than 5
                display_count = min(5, len(sorted_children))
                for j, child in enumerate(sorted_children[:display_count]):
                    # Calculate win rate properly considering perspective
                    raw_win_rate = child.value / child.visits if child.visits > 0 else 0
                    mcts_win_rate = 1.0 - raw_win_rate if root.state.turn != child.state.turn else raw_win_rate
                    
                    # Get win rate prediction for this move
                    next_state = state.clone()
                    next_state.apply_move(child.move)
                    predicted_win_rate = self.predict_win_rate(next_state)
                    
                    # Calculate win rate change
                    win_rate_change = predicted_win_rate - current_win_rate
                    change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
                    
                    print(f"{j+1}. ", end="")
                    debug_print_move_stats(
                        child.move, 
                        mcts_win_rate, 
                        predicted_win_rate, 
                        change_str, 
                        visits=child.visits, 
                        total_visits=root.visits,
                        is_capture=child.move.get("capture", False)
                    )
                    if j < display_count - 1:  # Add separator between moves, except after the last one
                        print("   ---")
                print("==================================\n")
            
            # Select the best child according to visits (standard MCTS approach)
            if not root.children:
                return None  # No valid moves
            
            # Standard MCTS selection: choose child with most visits
            best_child = max(root.children, key=lambda c: c.visits)
            
            # Calculate win rate properly considering perspective
            raw_win_rate = best_child.value / best_child.visits if best_child.visits > 0 else 0
            mcts_win_rate = 1.0 - raw_win_rate if root.state.turn != best_child.state.turn else raw_win_rate
            
            # Show the predicted win rate for the selected move
            next_state = state.clone()
            next_state.apply_move(best_child.move)
            predicted_win_rate = self.predict_win_rate(next_state)
            
            # Calculate win rate change
            win_rate_change = predicted_win_rate - current_win_rate
            change_str = f"+{win_rate_change:.4f}" if win_rate_change >= 0 else f"{win_rate_change:.4f}"
            
            if debug_config['selected_move']:
                print("\n========== SELECTED MOVE ==========")
                capture_text = "CAPTURE MOVE" if best_child.move.get("capture") else "regular move"
                print(f"Selected {capture_text} with {best_child.visits} visits ({best_child.visits/root.visits*100:.1f}% of total)")
                print(f"Move details: {best_child.move}")
                print(f"MCTS win rate: {mcts_win_rate:.4f}")
                print(f"Predicted win rate: {predicted_win_rate:.4f} ({change_str})")
                print(f"Tree reuse: {'YES - with prior knowledge' if tree_reused else 'NO - fresh search'}")
                print(f"Total iterations: {i}")
                print("==================================\n")
            
            # Store this tree's root for future reuse
            self.previous_root = root
            
            return best_child.move
            
        except Exception as e:
            # Log the error and fall back to a random move as a last resort
            print(f"MCTS error: {str(e)}. Falling back to random move.")
            # If we encounter an error, fall back to a random move
            valid_moves = state.get_valid_moves()
            if valid_moves:
                return random.choice(valid_moves)
            return None
            
    def _board_matches(self, board1, board2) -> bool:
        """Compare two game boards for equality"""
        for y in range(len(board1)):
            for x in range(len(board1[y])):
                cell1 = board1[y][x]
                cell2 = board2[y][x]
                
                # Compare None values
                if (cell1 is None) != (cell2 is None):
                    return False
                
                # Compare piece types
                if cell1 is not None and cell2 is not None:
                    if cell1["type"] != cell2["type"]:
                        return False
        
        return True
    
    def rollout(self, state: GameState) -> float:
        """Perform a rollout from the given state based on the selected policy."""
        # Store the initial state for normalization context
        self._last_state = state.clone()
        
        # Check if the state is already terminal
        if state.is_terminal():
            starting_player = state.turn
            winner = state.get_winner()
            
            if winner == "TIGER":
                return 1.0 if starting_player == "TIGER" else 0.0
            elif winner == "GOAT":
                return 0.0 if starting_player == "TIGER" else 1.0
            else:
                return 0.5  # Draw
        
        if self.rollout_policy == "random":
            return self._random_rollout(state)
        elif self.rollout_policy == "guided":
            return self._guided_rollout(state)
        elif self.rollout_policy == "lightweight":
            return self._lightweight_guided_rollout(state)
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
        
        # Check if we've reached a terminal state during rollout
        if current_state.is_terminal():
            winner = current_state.get_winner()
            
            if winner == "TIGER":
                return 1.0 if starting_player == "TIGER" else 0.0
            elif winner == "GOAT":
                return 0.0 if starting_player == "TIGER" else 1.0
            else:
                return 0.5  # Draw
        
        # Use win rate predictor at leaf node for non-terminal states
        tiger_win_rate = self.predict_win_rate(current_state)
        
        # Convert to starting player's perspective
        if starting_player == "TIGER":
            return tiger_win_rate
        else:
            return 1.0 - tiger_win_rate
    
    def _softmax(self, values, temperature):
        """
        Compute softmax probabilities with temperature scaling.
        
        Args:
            values: List of numerical values
            temperature: Temperature parameter (higher = more uniform distribution)
            
        Returns:
            List of probabilities that sum to 1.0
        """
        # Apply temperature scaling
        scaled = [v/temperature for v in values]
        
        # Subtract max for numerical stability
        max_val = max(scaled)
        exp_values = [math.exp(v - max_val) for v in scaled]
        
        # Return normalized probabilities
        sum_exp = sum(exp_values)
        return [v / sum_exp for v in exp_values]
        
    def _guided_rollout(self, state: GameState) -> float:
        """
        Optimized guided rollout using Softmax/Boltzmann selection.
        The guided_strictness parameter controls the temperature for move selection:
        - High strictness (close to 1.0) = low temperature = strongly prefer best moves
        - Low strictness (close to 0.0) = high temperature = more exploration
        
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
            
            # Always use softmax selection, with temperature controlled by strictness
            if len(valid_moves) > 1:
                # Convert strictness (0-1) to temperature (higher = more random)
                # With strictness=1.0, temperature=0.1 (very focused)
                # With strictness=0.0, temperature=10.0 (very random)
                temperature = 0.1 + (1.0 - self.guided_strictness) * 9.9
                
                # Evaluate all moves using cache when possible
                moves = []
                scores = []
                
                for move in valid_moves:
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
                    
                    moves.append(move)
                    scores.append(score)
                
                # Adjust scores based on player perspective
                if current_state.turn == "GOAT":
                    # For Goat, lower scores are better, so negate them
                    scores = [-s for s in scores]
                
                # Calculate selection probabilities using softmax
                probabilities = self._softmax(scores, temperature)
                
                # Sample a move according to these probabilities
                selected_move = random.choices(moves, weights=probabilities, k=1)[0]
            else:
                # Only one move available
                selected_move = valid_moves[0]
            
            current_state.apply_move(selected_move)
            depth += 1
        
        # Check if we've reached a terminal state during rollout
        if current_state.is_terminal():
            winner = current_state.get_winner()
            
            if winner == "TIGER":
                return 1.0 if starting_player == "TIGER" else 0.0
            elif winner == "GOAT":
                return 0.0 if starting_player == "TIGER" else 1.0
            else:
                return 0.5  # Draw
        
        # Use win rate predictor at leaf node for non-terminal states
        tiger_win_rate = self.predict_win_rate(current_state)
        
        # Convert to starting player's perspective
        if starting_player == "TIGER":
            return tiger_win_rate
        else:
            return 1.0 - tiger_win_rate

    def _lightweight_guided_rollout(self, state: GameState) -> float:
        """
        Lightweight guided rollout that uses simple strategic rules:
        - Tigers always take captures when available
        - Goats avoid immediate captures, with progressive bias toward exploring risky moves
        
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
            
            selected_move = None
            
            # Tiger's turn - prioritize captures
            if current_state.turn == "TIGER":
                # Find all capture moves
                capture_moves = [move for move in valid_moves if move.get("capture")]
                
                # If there are capture moves, select one randomly
                if capture_moves:
                    selected_move = random.choice(capture_moves)
                else:
                    selected_move = random.choice(valid_moves)
            
            # Goat's turn - avoid immediate captures with progressive bias
            else:  # GOAT's turn
                # Calculate how many empty cells remain (simple formula)
                empty_cells = 21 - current_state.goats_placed + current_state.goats_captured
                
                # Get the threatened nodes using the efficient method
                # Returns list of (x, y, landing_x, landing_y) tuples
                threatened_data = current_state.get_threatened_nodes()
                
                # Create a lookup for threatened positions to their landing positions
                # Format: {(x, y): (landing_x, landing_y)}
                threatened_lookup = {(x, y): (landing_x, landing_y) for x, y, landing_x, landing_y in threatened_data}
                
                # Categorize moves as safe or unsafe
                safe_moves = []
                unsafe_moves = []
                
                for move in valid_moves:
                    # PLACEMENT PHASE - simple check for immediate capture threat
                    if current_state.phase == "PLACEMENT":
                        target_x, target_y = move["x"], move["y"]
                        
                        # Check if this position is threatened
                        if (target_x, target_y) in threatened_lookup:
                            # Get the landing position
                            landing_x, landing_y = threatened_lookup[(target_x, target_y)]
                            
                            # Check if landing is empty (capture is possible)
                            if current_state.board[landing_y][landing_x] is None:
                                unsafe_moves.append(move)
                            else:
                                safe_moves.append(move)
                        else:
                            # Not threatened
                            safe_moves.append(move)
                    
                    # MOVEMENT PHASE - check both destination safety and if moving enables a capture
                    else:
                        from_x, from_y = move["from"]["x"], move["from"]["y"]
                        to_x, to_y = move["to"]["x"], move["to"]["y"]
                        
                        # Keep track of safety status
                        is_safe = True
                        
                        # Check if destination is threatened
                        if (to_x, to_y) in threatened_lookup:
                            landing_x, landing_y = threatened_lookup[(to_x, to_y)]
                            
                            # Check if landing is empty OR if this goat is moving from the landing position
                            is_landing_empty = current_state.board[landing_y][landing_x] is None
                            goat_from_landing = (from_x, from_y) == (landing_x, landing_y)
                            
                            if is_landing_empty or goat_from_landing:
                                is_safe = False
                        
                        # Check if emptying the source position enables a capture
                        # by finding any threatened positions that would land at our source position
                        for (x, y), (landing_x, landing_y) in threatened_lookup.items():
                            # If a tiger can capture a goat and land at our source position,
                            # then this move enables a capture
                            if (landing_x, landing_y) == (from_x, from_y):
                                # Additionally, check if the goat being captured exists
                                # x,y is the position of the captured goat
                                if current_state.board[y][x] is not None and current_state.board[y][x]["type"] == "GOAT":
                                    is_safe = False
                                    break
                        
                        # Categorize based on safety
                        if is_safe:
                            safe_moves.append(move)
                        else:
                            unsafe_moves.append(move)
                
                # Calculate progressive bias - as empty cells decrease, allow more exploration
                # Early game: almost always choose safe moves
                # Late game: consider unsafe moves more frequently
                # safe_bias = min(1.0, empty_cells / 21 + 0.2)
                
                # New formula - progressive bias that starts high and gradually decreases
                # Starts at 0.95 and can decrease to 0.75 by late game
                # This preserves the idea of trying risky moves more often as game progresses
                safe_bias = max(0.75, 0.95 - (0.20 * (1 - empty_cells / 21)))
                
                # Properly balance probabilities for individual moves across categories
                if safe_moves and unsafe_moves:
                    # Calculate per-move probabilities that sum to 1.0
                    
                    # Probability for any individual safe move
                    safe_move_prob = safe_bias / len(safe_moves)
                    
                    # Probability for any individual unsafe move
                    unsafe_move_prob = (1 - safe_bias) / len(unsafe_moves)
                    
                    # Combine moves with their probabilities
                    moves_with_probs = [(move, safe_move_prob) for move in safe_moves] + \
                                      [(move, unsafe_move_prob) for move in unsafe_moves]
                    
                    # Select a move using these individual probabilities
                    moves, probs = zip(*moves_with_probs)
                    selected_move = random.choices(moves, weights=probs, k=1)[0]
                elif safe_moves:
                    # Only safe moves available
                    selected_move = random.choice(safe_moves)
                elif unsafe_moves:
                    # Only unsafe moves available
                    selected_move = random.choice(unsafe_moves)
                else:
                    # No valid categorization (shouldn't happen but just in case)
                    selected_move = random.choice(valid_moves)
            
            # If we somehow didn't select a move, fall back to random
            if selected_move is None:
                selected_move = random.choice(valid_moves)
                
            current_state.apply_move(selected_move)
            depth += 1
        
        # Check if we've reached a terminal state during rollout
        if current_state.is_terminal():
            winner = current_state.get_winner()
            
            if winner == "TIGER":
                return 1.0 if starting_player == "TIGER" else 0.0
            elif winner == "GOAT":
                return 0.0 if starting_player == "TIGER" else 1.0
            else:
                return 0.5  # Draw
        
        # Use win rate predictor at leaf node for non-terminal states
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
        Handler function that calls either basic or advanced win rate predictor.
        Set use_advanced_predictor to True/False to switch between implementations.
        """
        # Toggle this flag to switch between predictors
        use_advanced_predictor = True  # Set to True when you want to use advanced
        
        if use_advanced_predictor:
            return self.predict_win_rate_advanced(state)
        else:
            return self.predict_win_rate_basic(state)
        
    def predict_win_rate_basic(self, state: GameState) -> float:
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
        elif goats_captured == 4:
            return 0.95 
        else:
            return 0.99

    def _map_captures_to_win_rate(self, capture_deficit: float, actual_captures: float) -> float:
        """
        Maps capture deficit to a non-linear win rate effect using a smooth sigmoid function
        that accounts for:
        1. How much better/worse than expected we're doing (deficit)
        2. How many captures remain until victory (5 - actual_captures)
        
        Args:
            capture_deficit: Effective captures - expected captures
            actual_captures: Actual number of goats captured (without adjustments)
            
        Returns:
            Win rate effect centered at 0 (to be added to 0.5)
        """
        # Handle negative deficits (goat advantage) by tracking sign
        is_negative = capture_deficit < 0
        deficit_abs = abs(capture_deficit)
        
        # Clamp actual captures to valid range [0, 4]
        # (5 captures is terminal state, not handled here)
        actual_captures = max(0, min(4, actual_captures))
        
        # Calculate remaining captures until victory
        remaining_captures = 5 - actual_captures
        
        # Clamp deficit to reasonable range based on remaining captures
        # No deficit should map higher than what would represent a win
        deficit_abs = min(remaining_captures, deficit_abs)
        
        # Maximum effect increases as more goats are captured
        # With 0 captures: max_effect = 0.49
        # With 4 captures: max_effect = 0.49 (same to avoid jumps)
        max_effect = 0.49
        
        # Steepness increases as more goats are captured
        # This creates a sharper curve when close to winning
        steepness_multiplier = 1.0 + (actual_captures / 4.0)
        
        # Base steepness determines how quickly the sigmoid rises
        # Higher = faster rise; tuned to match previous tiered approach
        base_steepness = 2.0
        steepness = base_steepness * steepness_multiplier
        
        # Midpoint shifts based on remaining captures
        # This creates different effective thresholds based on game stage
        # As more goats are captured, an increasing proportion of deficit
        # is needed to reach the midpoint
        midpoint_divisor = 4.0 - (2.0 * (actual_captures / 4.0))
        midpoint = remaining_captures / midpoint_divisor
        
        # Apply sigmoid function: max_effect / (1 + e^(-steepness * (x - midpoint)))
        sigmoid_value = max_effect / (1 + math.exp(-steepness * (deficit_abs - midpoint)))
        
        # Apply negative sign if original deficit was negative
        result = sigmoid_value if not is_negative else -sigmoid_value
        
        return result
        
    def _calculate_early_game_ratio(self, goats_placed: int, max_goats: int = 15) -> float:
        """Calculate the early game ratio ranging from 1.0 to 0.0 as goats_placed increases."""
        return max(0.0, 1.0 - (goats_placed / max_goats))
        
    def _calculate_dynamic_influence(self, effective_captures: float, goats_placed: int = 20) -> float:
        """
        Calculate capture influence that scales with both the number of effective captures
        and the game stage (goats placed).
        
        As captures approach 4 (one before terminal), the influence of captures increases.
        In very early game, positional heuristics are more important than captures.
        
        Args:
            effective_captures: The effective capture count (can be negative for goat advantage)
            goats_placed: Number of goats placed so far
        
        Returns:
            Influence percentage (dynamically adjusted based on game stage)
        """
        # Use absolute value to ensure proper scaling for both tiger and goat advantage
        capture_abs = min(4.0, max(0.0, abs(effective_captures)))
        
        # Early game ratio - ranges from 1.0 to 0.0 as goats placed goes from 0 to 15
        early_game_ratio = self._calculate_early_game_ratio(goats_placed)
        
        # Early game adjustment - reduce base influence significantly in early game
        # For first few goats, base should be around 0.5-0.6 instead of 0.85
        early_game_reduction = 0.35 * early_game_ratio  # Up to 0.35 reduction
        
        # Base and maximum influence percentages with early game adjustment
        base_influence = 0.85 - early_game_reduction
        max_influence = 0.99
        
        # Linear interpolation based on capture count
        return base_influence + (capture_abs / 4.0) * (max_influence - base_influence)

    def predict_win_rate_advanced(self, state: GameState) -> float:
        """
        Advanced win rate predictor that prioritizes captures and threats, with secondary
        consideration for positional factors.
        
        Returns a value from 0.0 to 1.0 from Tiger's perspective:
        - 0.0: Certain Goat win
        - 0.5: Balanced game
        - 1.0: Certain Tiger win
        """
        # Handle terminal states
        if state.is_terminal():
            winner = state.get_winner()
            if winner == "TIGER": return 1.0  # Definitive Tiger win
            elif winner == "GOAT": return 0.0  # Definitive Goat win
            else: return 0.5  # Draw
        
        # Get actual captures
        goats_captured = state.goats_captured
            
        # Get all tiger moves for threat calculation
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Get the threatened value and log the max value found
        threatened_value = self.minimax_agent._count_threatened_goats(all_tiger_moves, state.turn)
        
        # Get closed spaces data
        closed_regions = self.minimax_agent._count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Count movable tigers
        movable_tigers_count = self.minimax_agent._count_movable_tigers(all_tiger_moves)
        
        # Calculate closed space impact on effective captures
        # Scale from 0.3 at move 1, to 0.8 by move 15, then to 0.95 by move 20
        if state.goats_placed <= 1:
            closed_space_factor = 0.3
        elif state.goats_placed <= 15:
            # Linear interpolation between 0.3 and 0.8 as goats_placed goes from 1 to 15
            closed_space_factor = 0.3 + (0.5 * (state.goats_placed - 1) / 14)
        elif state.goats_placed >= 20:
            closed_space_factor = 0.95
        else:
            # Linear interpolation between 0.8 and 0.95 as goats_placed goes from 15 to 20
            closed_space_factor = 0.8 + (0.15 * (state.goats_placed - 15) / 5)
        
        # Calculate the closed space value by multiplying the factor with total closed spaces
        closed_space_value = total_closed_spaces * closed_space_factor
        
        # Add threatened value and subtract closed_space_value to get effective_captures
        effective_captures = goats_captured + threatened_value - closed_space_value
        
        # More radical mapping: 0.1 -> 0.95 for captures 0-4
        # This uses the full range as requested
        capture_impact = (0.95 - 0.1) / 4.0  # 0.2125 per capture
        
        # Cap effective captures at 4 for win rate calculation
        capped_effective_captures = min(4.0, max(0.0, effective_captures))
        capture_base_rate = 0.1 + (capped_effective_captures * capture_impact)
        
        # Reserve much smaller space for positional heuristics
        # Maximum positional contribution reduced to 5% of a single capture's impact
        # This ensures captures remain even more dominant
        max_position_contribution = capture_impact * 0.05
        
        # Calculate all positional factors used in minimax evaluation
        position_score = self.minimax_agent._calculate_tiger_positional_score(state)
        spacing_score = self.minimax_agent._calculate_tiger_optimal_spacing(state)
        edge_score = self.minimax_agent._calculate_goat_edge_preference(state)
        
        # Normalize movable tigers count to 0-1 range (total tigers is 4)
        normalized_movable_tigers = movable_tigers_count / 4.0
        
        # Combine positional factors with appropriate weights
        position_weight = 0.3
        spacing_weight = 0.3
        edge_weight = 0.2
        movable_tigers_weight = 0.2  # Add weight for movable tigers
        
        # Combine scores (invert edge score as it favors goats)
        position_combined = (
            position_score * position_weight +
            spacing_score * spacing_weight - 
            edge_score * edge_weight +  # Inverted because higher edge score favors goats
            normalized_movable_tigers * movable_tigers_weight  # Add movable tigers factor
        )
        
        # Scale the positional contribution to fit within the allowed budget
        # Normalize to [-1, 1] range first
        normalized_position = max(-1.0, min(1.0, position_combined))
        # Scale to fit within budget
        position_contribution = normalized_position * max_position_contribution
        
        # Combine base rate with position contribution
        win_rate = capture_base_rate + position_contribution
        
        # Ensure valid range
        return max(0.01, min(0.99, win_rate))