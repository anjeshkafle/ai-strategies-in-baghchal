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
            'move_analysis': True,    # Show detailed analysis of possible moves
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
        Advanced win rate predictor that combines multiple heuristics:
        1. Captures (raw and with adjustments)
        2. Tiger positional score
        3. Tiger optimal spacing
        4. Goat edge preference
        
        Returns a value from 0.0 to 1.0 from Tiger's perspective:
        - 0.0: Certain Goat win
        - 0.5: Balanced game
        - 1.0: Certain Tiger win
        """
        # Short-circuit terminal states
        if state.is_terminal():
            winner = state.get_winner()
            if winner == "TIGER":
                return 0.99  # Near certain Tiger win
            elif winner == "GOAT":
                return 0.01  # Near certain Goat win
            else:
                return 0.5   # Draw
        
        # STEP 1: Calculate expected captures based on game progression
        # Track game state
        goats_placed = state.goats_placed
        goats_captured = state.goats_captured
        
        # CRITICAL FIX: Ensure win rate increases monotonically with actual captures
        # Use actual captures as the primary determinant
        # Create a base win rate that increases with captures (0.5 → 0.99 as captures go from 0 → 5)
        # This ensures captures always have a strong positive effect
        capture_base_rate = 0.5 + (goats_captured / 5.0) * 0.49
        
        # Calculate expected captures at this stage - using smooth function
        if goats_placed < 15:
            # Sigmoid function that stays close to 0 until approaching 15 goats
            transition_steepness = 0.5
            expected_captures = 2.0 / (1 + math.exp(-transition_steepness * (goats_placed - 15)))
        else:
            # After 15 goats, linear progression (2/5 of additional goats)
            expected_captures = (goats_placed - 15) * (2 / 5)
        
        # STEP 2: Calculate effective future captures using key heuristics
        # Start with actual captures
        effective_captures = goats_captured
        
        # Get all tiger moves (needed for heuristics calculation)
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Closed spaces heuristic (more closed spaces favor goats)
        closed_regions = self.minimax_agent._count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Apply a smooth sigmoid-based scaling for closed spaces
        # This avoids sharp transitions in impact between early and late game
        min_closed_space_factor = 0.3
        max_closed_space_factor = 0.8
        factor_range = max_closed_space_factor - min_closed_space_factor
        
        # Sigmoid function: scale from min to max based on placement progress
        # Centered at 10 goats (midpoint of placement phase)
        placement_sigmoid = 1.0 / (1 + math.exp(-0.4 * (goats_placed - 10)))
        closed_space_factor = min_closed_space_factor + (factor_range * placement_sigmoid)
        
        # Calculate closed space value using the smooth factor
        closed_space_value = total_closed_spaces * closed_space_factor
        
        # Subtract closed space effect (negative contribution to effective captures)
        effective_captures -= closed_space_value
        
        # Threatened goats (using updated function that returns normalized value)
        threatened_value = self.minimax_agent._count_threatened_goats(all_tiger_moves, state.turn)
        
        # Add threatened value to effective captures
        effective_captures += threatened_value
        
        # STEP 3: Calculate capture deficit (actual+potential - expected)
        capture_deficit = effective_captures - expected_captures
        
        # STEP 4: Calculate additional positional heuristics with dynamic equilibrium points
        
        # Tiger positional score (normalized 0-1, higher is better for tiger)
        position_score = self.minimax_agent._calculate_tiger_positional_score(state)
        
        # Calculate dynamic equilibrium point for tiger position score using sigmoid
        # Smooth transition from 0.5 to 0.33 centered around 12.5 goats placed
        position_max = 0.5
        position_min = 0.33
        position_range = position_max - position_min
        position_transition_steepness = 0.4  # Controls how quickly the transition happens
        position_transition_midpoint = 12.5  # Centered between early and late game
        
        # Sigmoid transition: starts at position_max, transitions to position_min
        position_transition_factor = 1.0 / (1 + math.exp(-position_transition_steepness * (goats_placed - position_transition_midpoint)))
        position_equilibrium = position_max - (position_range * position_transition_factor)
            
        # Convert using dynamic equilibrium point (centered at 0)
        position_factor = position_score - position_equilibrium
        
        # Tiger optimal spacing score (normalized 0-1, higher is better for tiger)
        spacing_score = self.minimax_agent._calculate_tiger_optimal_spacing(state)
        
        # Calculate dynamic equilibrium point for optimal spacing score using same sigmoid
        # Uses same parameters as position score for consistency
        spacing_equilibrium = position_equilibrium  # Same equilibrium pattern as position score
            
        # Convert using dynamic equilibrium point (centered at 0)
        spacing_factor = spacing_score - spacing_equilibrium
        
        # Goat edge preference (normalized 0-1, higher is better for goat)
        edge_score = self.minimax_agent._calculate_goat_edge_preference(state)
        
        # Calculate dynamic equilibrium for edge preference using smooth sigmoid transitions
        # Early phase: transition from 1.0 to 0.8 (centered at 3 goats)
        # Late phase: transition from 0.8 to 0.1 (centered at 16 goats)
        
        # Parameters for early transition (1.0 to 0.8)
        early_max = 1.0
        early_min = 0.8
        early_range = early_max - early_min
        early_steepness = 0.8  # Steeper for a faster early transition
        early_midpoint = 3.0  # Centered early in placement phase
        
        # Parameters for late transition (0.8 to 0.1)
        late_max = 0.8
        late_min = 0.1
        late_range = late_max - late_min
        late_steepness = 0.3  # Gentler for a smoother late transition
        late_midpoint = 16.0  # Centered later in game
        
        # Early transition factor (0 to 1)
        early_transition = 1.0 / (1 + math.exp(-early_steepness * (goats_placed - early_midpoint)))
        
        # Late transition factor (0 to 1)
        late_transition = 1.0 / (1 + math.exp(-late_steepness * (goats_placed - late_midpoint)))
        
        # Calculate equilibrium based on both transitions
        # First apply early transition (1.0 to 0.8)
        edge_equilibrium = early_max - (early_range * early_transition)
        
        # Then apply late transition (0.8 to 0.1) which takes effect later
        edge_equilibrium = edge_equilibrium - (late_range * late_transition * early_transition)
            
        # Convert using dynamic equilibrium point (centered at 0, inverted since higher edge score favors goats)
        edge_factor = edge_equilibrium - edge_score
        
        # STEP 5: Non-linear mapping of capture deficit to win rate effect
        capture_effect = self._map_captures_to_win_rate(capture_deficit, goats_captured)
        
        # STEP 6: Calculate dynamic influence based on effective captures
        capture_influence = self._calculate_dynamic_influence(effective_captures, goats_placed)
        
        # STEP 7: Combine components with dynamic weighting
        # Capture component with non-linear effect and dynamic influence
        capture_component = 0.5 + capture_effect * capture_influence
        
        # Additional heuristics with remaining influence (weighted according to minimax proportions)
        heuristic_influence = 1.0 - capture_influence
        
        # Calculate early and late game ratios for dynamic weights
        early_game_ratio = self._calculate_early_game_ratio(goats_placed)  # Decreases from 1.0 to 0.0 as goats_placed goes from 0 to 15
        late_game_ratio = min(1, goats_placed / 15)         # Increases from 0.0 to 1.0 as goats_placed goes from 0 to 15
        
        # Dynamic weights similar to minimax agent:
        # Position weight is more important in early game
        position_weight_factor = 1.0 + (0.5 * early_game_ratio)  # Ranges from 1.0 to 1.5
        # Edge weight is more important in early game
        edge_weight_factor = 1.0 + (1.0 * early_game_ratio)      # Ranges from 1.0 to 2.0
        # Spacing weight increases in mid-to-late game
        spacing_weight_factor = 1.0 + (0.7 * late_game_ratio)    # Ranges from 1.0 to 1.7
        
        # Base weight proportions
        base_position_weight = 1.0 / 5.5
        base_spacing_weight = 1.5 / 5.5
        base_edge_weight = 3.0 / 5.5
        
        # Apply dynamic factors to weights
        position_weight = base_position_weight * position_weight_factor
        spacing_weight = base_spacing_weight * spacing_weight_factor
        
        # Amplify edge weight significantly for early placement phase (first 5 goats)
        early_placement_boost = max(0.0, 3.0 * (1.0 - min(1.0, goats_placed / 5)))
        edge_weight = base_edge_weight * (edge_weight_factor + early_placement_boost)
        
        # Apply weighted factors
        heuristic_sum = (
            (position_factor * position_weight) + 
            (spacing_factor * spacing_weight) + 
            (edge_factor * edge_weight)
        )
        heuristic_component = 0.5 + heuristic_sum * heuristic_influence
        
        # CRITICAL FIX: Blend the capture base rate (which guarantees monotonic increases with captures)
        # with the heuristic-influenced rate, with more weight to the base rate as captures increase
        capture_importance = min(1.0, goats_captured / 4.0)  # Increases from 0 to 1 as captures go from 0 to 4
        blended_rate = (capture_base_rate * capture_importance) + (
            (0.5 + ((capture_component - 0.5) + (heuristic_component - 0.5))) * (1.0 - capture_importance)
        )
        
        # Ensure win rate stays within valid range
        return max(0.01, min(0.99, blended_rate))