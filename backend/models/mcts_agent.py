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
        """
        Select a child node using the UCB formula with progressive bias for captures.
        
        The progressive bias adds a term to the UCB formula that diminishes with
        increased visits, providing initial guidance while preserving asymptotic behavior.
        """
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def ucb_score(child):
            # Avoid division by zero
            if child.visits == 0:
                return float('inf')  # Ensures unvisited children are explored first
                
            # Calculate win rate (exploitation term)
            win_rate = child.value / child.visits
            
            # IMPORTANT: win_rate is already from the child's player perspective
            # We need to convert it to the current (parent) player's perspective
            # If the players are different, we need to invert the win rate
            if self.state.turn != child.state.turn:
                # Invert win rate when switching players
                win_rate = 1.0 - win_rate
                
            # Standard UCB exploration term
            exploration = exploration_weight * math.sqrt(log_visits / child.visits)
            
            # Progressive bias for capture moves
            # This bias diminishes as visits increase, preserving asymptotic UCB behavior
            progressive_bias = 0.0
            if self.state.turn == "TIGER" and child.move and child.move.get("capture"):
                # Add a bias for Tiger capture moves that diminishes with visits
                # This is mathematically sound and preserves UCB properties
                bias_weight = 100.0  # Significant initial weight
                progressive_bias = bias_weight / (child.visits + 1)  # Diminishes with visits
            
            # Combined UCB formula with progressive bias
            return win_rate + exploration + progressive_bias
        
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


class MCTSAgent:
    """Monte Carlo Tree Search agent for the Bagh Chal game."""
    
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
                while node is not None:
                    node.update(result)
                    node = node.parent
                    
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
                    mcts_win_rate = child.value / child.visits if child.visits > 0 else 0
                    
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
                for child, mcts_win_rate in capture_moves:
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
            mcts_win_rate = best_child.value / best_child.visits if best_child.visits > 0 else 0
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
        Perform a completely random rollout.
        
        Returns a value in [0.0, 1.0] where:
        - 1.0 means win for the CURRENT player at the start of the rollout
        - 0.0 means loss for the CURRENT player at the start of the rollout
        - 0.5 means draw
        """
        import time
        start_time = time.time()
        rollout_timeout = 0.5  # Maximum time for a single rollout in seconds
        
        starting_player = state.turn  # Remember who started the rollout
        current_state = state.clone()
        max_depth = self.max_rollout_depth
        depth = 0
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Check timeout to prevent infinite loops
            if depth % 5 == 0 and time.time() - start_time > rollout_timeout:
                # If timeout, get win rate from Tiger's perspective
                tiger_win_rate = self.predict_win_rate(current_state)
                
                # Convert to starting player's perspective
                if starting_player == "TIGER":
                    return tiger_win_rate  # Already in Tiger's perspective
                else:
                    return 1.0 - tiger_win_rate  # Convert to Goat's perspective
            
            # Only check for repetition in movement phase
            if current_state.phase == "MOVEMENT":
                # Create a hash of the current state
                state_hash = self._get_state_hash(current_state)
                
                # Check for threefold repetition
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    return 0.5  # Draw due to threefold repetition
            
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            move = random.choice(valid_moves)
            current_state.apply_move(move)
            depth += 1
        
        # If we hit max depth, use win rate predictor
        if depth >= max_depth:
            # Get win rate from Tiger's perspective
            tiger_win_rate = self.predict_win_rate(current_state)
            
            # Convert to starting player's perspective
            if starting_player == "TIGER":
                return tiger_win_rate  # Already in Tiger's perspective
            else:
                return 1.0 - tiger_win_rate  # Convert to Goat's perspective
        
        # Otherwise score based on winner
        winner = current_state.get_winner()
        
        # Return from starting player's perspective
        if winner == starting_player:
            return 1.0  # Starting player won
        elif winner is None:
            return 0.5  # Draw
        else:
            return 0.0  # Starting player lost
    
    def _guided_rollout(self, state: GameState) -> float:
        """
        Perform a rollout guided by evaluation function with controllable strictness.
        
        Returns a value in [0.0, 1.0] where:
        - 1.0 means win for the CURRENT player at the start of the rollout
        - 0.0 means loss for the CURRENT player at the start of the rollout
        - 0.5 means draw
        
        The strictness parameter (0.0 to 1.0) controls how deterministic the rollout is:
        - 0.0: Fully probabilistic selection based on evaluation scores
        - 1.0: Always selects the best evaluated move
        - Values between: Increasingly favor the best moves
        """
        starting_player = state.turn  # Remember who started the rollout
        current_state = state.clone()
        max_depth = self.max_rollout_depth
        depth = 0
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Only check for repetition in movement phase
            if current_state.phase == "MOVEMENT":
                # Create a hash of the current state
                state_hash = self._get_state_hash(current_state)
                
                # Check for threefold repetition
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    return 0.5  # Draw due to threefold repetition
            
            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                break
            
            # Check if there are any capture moves available
            capture_moves = [move for move in valid_moves if move.get("capture")]
            
            # If Tiger has capture moves available, prioritize them
            if current_state.turn == "TIGER" and capture_moves:
                # Either select deterministically or probabilistically based on strictness
                if self.guided_strictness >= 0.7:  # High strictness = always capture
                    # Get minimax scores for capture moves to pick the best one
                    capture_scores = []
                    for move in capture_moves:
                        next_state = current_state.clone()
                        next_state.apply_move(move)
                        score = self.minimax_agent.evaluate(next_state)
                        capture_scores.append((move, score))
                    
                    # Select the best capture move
                    selected_move = max(capture_scores, key=lambda x: x[1])[0]
                else:
                    # Select a random capture move with some bias toward better ones
                    selected_move = random.choice(capture_moves)
            else:
                # Get base evaluation of current state
                base_eval = self.minimax_agent.evaluate(current_state)
                
                # Calculate delta (improvement) for each move
                move_deltas = []
                for move in valid_moves:
                    next_state = current_state.clone()
                    next_state.apply_move(move)
                    
                    # Get raw score from minimax
                    next_eval = self.minimax_agent.evaluate(next_state)
                    
                    # Apply a capture bonus for Tiger moves
                    if current_state.turn == "TIGER" and move.get("capture"):
                        next_eval *= 1.5  # Significant bonus for captures
                    
                    # Calculate delta (improvement from current position)
                    delta = next_eval - base_eval
                    
                    # Store the move and its delta
                    move_deltas.append((move, delta))
                
                # Select move based on strictness parameter
                if self.guided_strictness >= 1.0:
                    # Fully deterministic: always select the best move
                    if current_state.turn == "TIGER":
                        # For Tiger: higher delta is better
                        selected_move = max(move_deltas, key=lambda x: x[1])[0]
                    else:
                        # For Goat: lower delta is better (but avoid suicide moves)
                        # First, identify moves that lead to immediate capture
                        goat_moves_with_risk = []
                        for move, delta in move_deltas:
                            next_state = current_state.clone()
                            next_state.apply_move(move)
                            
                            # Check if this move allows Tiger to capture immediately
                            tiger_moves = get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER")
                            has_capture = any(m.get("capture") for m in tiger_moves)
                            goat_moves_with_risk.append((move, delta, has_capture))
                        
                        # First try to find the best move without immediate capture risk
                        safe_moves = [(move, d) for move, d, has_risk in goat_moves_with_risk if not has_risk]
                        if safe_moves:
                            selected_move = min(safe_moves, key=lambda x: x[1])[0]
                        else:
                            # If all moves are risky, just pick the best one
                            selected_move = min(move_deltas, key=lambda x: x[1])[0]
                else:
                    # Probabilistic selection with strictness-based contrast
                    # Higher strictness = more emphasis on better moves
                    
                    # Calculate power factor based on strictness
                    # At strictness=0, power=1 (no change to scores)
                    # At strictness=0.9, power=10 (extreme contrast)
                    power = 1.0 + (self.guided_strictness * 9.0)  # Ranges from 1.0 to 10.0
                    
                    # Identify capture moves and non-capture moves
                    capture_moves = [(move, delta) for move, delta in move_deltas if move.get("capture")]
                    non_capture_moves = [(move, delta) for move, delta in move_deltas if not move.get("capture")]
                    
                    # Apply different adjustment for Tiger and Goat
                    if current_state.turn == "TIGER":
                        # For Tiger: favor capture moves strongly, but still consider good non-captures
                        if capture_moves:
                            # Boost capture moves' scores
                            adjusted_scores = [(move, max(0, delta) * 2.0 + 0.1) for move, delta in capture_moves]
                            # Include non-captures with lower weight
                            adjusted_scores += [(move, max(0, delta) + 0.1) for move, delta in non_capture_moves]
                        else:
                            # No captures available, use standard adjustment
                            adjusted_scores = [(move, max(0, delta) + 0.1) for move, delta in move_deltas]
                    else:
                        # For Goat: strongly avoid moves that allow Tiger to capture
                        adjusted_scores = []
                        for move, delta in move_deltas:
                            next_state = current_state.clone()
                            next_state.apply_move(move)
                            
                            # Check if this move allows Tiger to capture immediately
                            tiger_moves = get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER")
                            has_capture = any(m.get("capture") for m in tiger_moves)
                            
                            if has_capture:
                                # Heavily penalize moves that lead to immediate capture
                                score = max(0, -delta) * 0.1 + 0.01  # Very small score
                            else:
                                # Favor moves that don't allow capture
                                score = max(0, -delta) + 0.1
                            
                            adjusted_scores.append((move, score))
                    
                    # Apply the power function to increase contrast between scores
                    adjusted_scores = [(move, score**power) for move, score in adjusted_scores]
                    
                    # Calculate probabilities proportional to adjusted scores
                    total_score = sum(score for _, score in adjusted_scores)
                    if total_score > 0:  # Protect against division by zero
                        probabilities = [score/total_score for _, score in adjusted_scores]
                        
                        # Select move based on probabilities (weighted random)
                        moves = [move for move, _ in adjusted_scores]
                        selected_move = random.choices(moves, weights=probabilities, k=1)[0]
                    else:
                        # Fallback to random if all scores are zero
                        selected_move = random.choice(valid_moves)
            
            current_state.apply_move(selected_move)
            depth += 1
        
        # If we hit max depth, use win rate predictor
        if depth >= max_depth:
            # Get win rate from Tiger's perspective
            tiger_win_rate = self.predict_win_rate(current_state)
            
            # Convert to starting player's perspective
            if starting_player == "TIGER":
                return tiger_win_rate  # Already in Tiger's perspective
            else:
                return 1.0 - tiger_win_rate  # Convert to Goat's perspective
        
        # Otherwise score based on winner
        winner = current_state.get_winner()
        
        # Return from starting player's perspective
        if winner == starting_player:
            return 1.0  # Starting player won
        elif winner is None:
            return 0.5  # Draw
        else:
            return 0.0  # Starting player lost

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
        Simplified win rate predictor that focuses primarily on captures and avoids sudden jumps.
        
        Returns values from Tiger's perspective:
        - 0.0: Strong goat advantage (goat likely to win)
        - 0.5: Balanced position (draw likely)
        - 1.0: Strong tiger advantage (tiger likely to win)
        
        For Tiger player, higher values are better.
        For Goat player, lower values are better.
        """
        # Get basic game state information
        goats_placed = state.goats_placed
        goats_captured = state.goats_captured
        current_turn = state.turn
        
        # Check for capture opportunities for tiger
        tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        threatened_goats = self.minimax_agent._count_threatened_goats(tiger_moves)
        
        # If it's Tiger's turn and there are threatened goats, we consider this capture "certain"
        # This effectively means we calculate the win rate as if the capture has already happened
        effective_captures = goats_captured
        if current_turn == "TIGER" and threatened_goats > 0:
            effective_captures += 1
        
        # Start with a balanced position
        win_rate = 0.5
        
        # Adjust win rate based on game phase and captures
        if state.phase == "PLACEMENT":
            # Early placement phase (0-10 goats)
            if goats_placed <= 10:
                if effective_captures == 0:
                    # Gradual disadvantage for tiger as goats are placed
                    win_rate = 0.5 - (goats_placed * 0.005)  # Small gradual decrease
                elif effective_captures == 1:
                    win_rate = 0.8  # First capture gives major advantage
                elif effective_captures == 2:
                    win_rate = 0.85  # Second capture increases advantage slightly
                else:  # 3+ captures
                    win_rate = 0.9  # Strong tiger advantage
                    
            # Mid-placement (11-15 goats)
            elif goats_placed <= 15:
                if effective_captures == 0:
                    # Gradual advantage for goats
                    win_rate = 0.45 - ((goats_placed - 10) * 0.01)  # More goats = better for goats
                elif effective_captures == 1:
                    win_rate = 0.7  # First capture still gives advantage but less than early game
                elif effective_captures == 2:
                    win_rate = 0.8  # Second capture gives strong advantage
                else:  # 3+ captures
                    win_rate = 0.85  # Very strong tiger advantage
                    
            # Late placement (16-20 goats)
            else:
                if effective_captures == 0:
                    # Strong goat advantage without captures
                    win_rate = 0.35 - ((goats_placed - 15) * 0.01)  # More goats = better for goats
                elif effective_captures == 1:
                    win_rate = 0.6  # First capture balances the game
                elif effective_captures == 2:
                    win_rate = 0.75  # Second capture gives tiger advantage
                elif effective_captures == 3:
                    win_rate = 0.85  # Third capture gives strong advantage
                else:  # 4+ captures
                    win_rate = 0.9   # Very strong tiger advantage
        else:  # MOVEMENT phase
            # Base win rate depends on captures
            if effective_captures == 0:
                win_rate = 0.3       # Strong goat advantage 
            elif effective_captures == 1:
                win_rate = 0.4       # Moderate goat advantage 
            elif effective_captures == 2:
                win_rate = 0.55      # Balanced with slight tiger advantage
            elif effective_captures == 3:
                win_rate = 0.7       # Strong tiger advantage
            elif effective_captures == 4:
                win_rate = 0.85      # Very strong tiger advantage
            else:  # 5 captures (tiger win)
                win_rate = 1.0       # Tiger victory
        
        # Additional adjustment for non-immediate capture opportunities
        # This rewards positions where Tiger has captures available, even if not Tiger's turn
        if threatened_goats > 0 and current_turn != "TIGER":
            win_rate += 0.05 * threatened_goats  # Bonus scales with threat level
        
        # Clamp final win rate to valid range [0.0, 1.0]
        return max(0.0, min(1.0, win_rate)) 