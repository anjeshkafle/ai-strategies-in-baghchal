from typing import List, Dict, Optional, Tuple
import random
import math
from models.game_state import GameState
from models.minimax_agent import MinimaxAgent

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
        Select a child node using an enhanced UCB formula that better values
        promising moves while maintaining exploration balance.
        """
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def enhanced_ucb_score(child):
            # Avoid division by zero
            if child.visits == 0:
                return float('inf')  # Ensures unvisited children are explored first
                
            # Calculate win rate (exploitation term)
            win_rate = child.value / child.visits
            
            # If it's Goat's turn at parent, invert the win rate for correct maximization
            if self.state.turn == "GOAT":
                win_rate = 1.0 - win_rate
                
            # Standard UCB exploration term
            exploration = exploration_weight * math.sqrt(log_visits / child.visits)
            
            # Enhanced UCB formula - progressive bias toward higher win rates
            # Higher exponent (>1) for win_rate increases focus on promising paths
            # This naturally rewards high-value capture moves without explicit bias
            ucb = (win_rate ** 1.2) + exploration
            
            # Add a small bonus for recent visits to balance exploration/exploitation
            recency_bonus = 0.01 * (child.visits / self.visits) if self.visits > 0 else 0
            
            return ucb + recency_bonus
        
        return max(self.children, key=enhanced_ucb_score)
    
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
                 rollout_policy: str = "random", use_minimax_eval: bool = True):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rollout_policy = rollout_policy  # "random", "weighted", or "guided"
        self.use_minimax_eval = use_minimax_eval
        
        # Create a minimax agent for evaluation if needed
        if use_minimax_eval:
            self.minimax_agent = MinimaxAgent(max_depth=2)  # Shallow depth for quick evals
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state using MCTS."""
        try:
            import time
            start_time = time.time()
            max_time_seconds = 5  # Maximum time to spend on calculation
            
            # Check for edge cases first
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                return None
                
            # If there's only one valid move, return it without running MCTS
            if len(valid_moves) == 1:
                print(f"Only one valid move available, returning immediately")
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
                print("\nMCTS Node Statistics:")
                print("=====================")
                # Sort children by visits for display
                sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
                for i, child in enumerate(sorted_children):
                    capture_str = " (CAPTURE)" if child.move.get("capture") else ""
                    win_rate = child.value / child.visits if child.visits > 0 else 0
                    print(f"{i+1}. Move: {child.move}{capture_str}")
                    print(f"   Visits: {child.visits}, Value: {child.value:.2f}, Win rate: {win_rate:.2f}")
                    
                    # Evaluate move with minimax for comparison
                    if self.use_minimax_eval:
                        minimax_score = self.evaluate_move(state, child.move)
                        print(f"   Minimax evaluation: {minimax_score}")
                print("=====================\n")
            
            # Select the best move using a robust approach that balances win rate and visits
            if not root.children:
                return None  # No valid moves

            # Calculate scores for move selection
            children_with_scores = []
            capture_children = []
            best_win_rate = 0.0
            
            for child in root.children:
                # Skip unvisited children
                if child.visits == 0:
                    continue
                
                # Calculate win rate and visit statistics
                win_rate = child.value / child.visits
                visit_ratio = child.visits / max(1, root.visits)
                
                # Track best win rate for reference
                best_win_rate = max(best_win_rate, win_rate)
                
                # Check if move is a capture
                is_capture = child.move.get("capture", False)
                if is_capture:
                    capture_children.append((child, win_rate, visit_ratio))
                
                # Our robust score heavily weights win rate (90%) over visit count (10%)
                # Win rate is squared to emphasize high-value moves even more
                robust_score = (0.9 * win_rate**2) + (0.1 * visit_ratio)
                children_with_scores.append((child, robust_score, win_rate, visit_ratio, is_capture))
            
            # Sort by robust score for better logging
            sorted_children = sorted(children_with_scores, key=lambda x: x[1], reverse=True)
            
            # Print top candidates by robust score
            print("\nTop candidates by robust score:")
            for i, (child, score, win_rate, visit_ratio, is_capture) in enumerate(sorted_children[:3]):
                move_type = "CAPTURE" if is_capture else "regular"
                print(f"  {i+1}. {move_type} move: visits={child.visits} ({visit_ratio:.2f}), win_rate={win_rate:.2f}, score={score:.3f}")
            
            # Capture move selection logic:
            # If there's a capture move with a win rate very close to the best win rate,
            # prefer it naturally without introducing explicit bias
            if capture_children:
                for child, win_rate, visit_ratio in capture_children:
                    # If capture move has a reasonably comparable win rate, select it
                    if win_rate >= best_win_rate * 0.97 and child.visits >= 5:
                        best_child = child
                        print(f"\nSelected CAPTURE MOVE with win rate {win_rate:.2f} (best win rate: {best_win_rate:.2f})")
                        print(f"Visits: {child.visits} ({visit_ratio*100:.1f}% of total)")
                        return best_child.move
            
            # Otherwise, select based on robust score
            if children_with_scores:
                best_child, best_score = max(children_with_scores, key=lambda x: x[1])[:2]
                
                # Output details about the selection
                win_rate = best_child.value / best_child.visits
                capture_text = "CAPTURE MOVE" if best_child.move.get("capture") else "regular move"
                print(f"\nSelected {capture_text} with {best_child.visits} visits ({best_child.visits/root.visits*100:.1f}% of total)")
                print(f"Win rate: {win_rate:.2f}, Robust score: {best_score:.2f}")
                
                return best_child.move
            else:
                # Fallback to standard MCTS selection if all children have 0 visits
                best_child = max(root.children, key=lambda c: c.visits)
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
        if self.rollout_policy == "random":
            return self._random_rollout(state)
        elif self.rollout_policy == "weighted":
            return self._weighted_rollout(state)
        elif self.rollout_policy == "guided":
            return self._guided_rollout(state)
        else:
            return self._random_rollout(state)  # Default to random
    
    def _random_rollout(self, state: GameState) -> float:
        """Perform a completely random rollout."""
        import time
        start_time = time.time()
        rollout_timeout = 0.5  # Maximum time for a single rollout in seconds
        
        current_state = state.clone()
        max_depth = 30  # Reduced from 50 to prevent long simulations
        depth = 0
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Check timeout to prevent infinite loops
            if depth % 5 == 0 and time.time() - start_time > rollout_timeout:
                # If timeout, use evaluation function
                if self.use_minimax_eval:
                    eval_score = self.minimax_agent.evaluate(current_state)
                    # The evaluation function returns higher values for Tiger advantage
                    # We need to normalize to [0,1] where 1 means Tiger win
                    return self._normalize_eval_score(eval_score)
                else:
                    return 0.5  # Draw if no evaluation function
            
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
        
        # If we hit max depth, use evaluation function
        if depth >= max_depth and self.use_minimax_eval:
            eval_score = self.minimax_agent.evaluate(current_state)
            return self._normalize_eval_score(eval_score)
        
        # Otherwise score based on winner
        winner = current_state.get_winner()
        if winner == "TIGER":
            return 1.0
        elif winner == "GOAT":
            return 0.0
        else:
            return 0.5  # Draw
    
    def _weighted_rollout(self, state: GameState) -> float:
        """Perform a rollout with probability-weighted move selection."""
        current_state = state.clone()
        max_depth = 50
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
                
            # If using minimax evaluation
            if self.use_minimax_eval:
                # Evaluate all moves using minimax
                move_scores = []
                for move in valid_moves:
                    next_state = current_state.clone()
                    next_state.apply_move(move)
                    score = self.minimax_agent.evaluate(next_state)
                    
                    # Adjust score based on player
                    if current_state.turn == "GOAT":
                        score = -score  # Invert for goat (lower is better)
                        
                    # Ensure all scores are positive for probability calculation
                    score = max(score, 0) + 1  # Add 1 to avoid zero probabilities
                    move_scores.append((move, score))
                
                # Calculate probabilities
                total_score = sum(score for _, score in move_scores)
                probabilities = [score/total_score for _, score in move_scores]
                
                # Select move based on probabilities
                moves = [move for move, _ in move_scores]
                selected_move = random.choices(moves, weights=probabilities, k=1)[0]
            else:
                # Fall back to random selection if not using minimax
                selected_move = random.choice(valid_moves)
                
            current_state.apply_move(selected_move)
            depth += 1
        
        # If we hit max depth, use evaluation function
        if depth >= max_depth and self.use_minimax_eval:
            eval_score = self.minimax_agent.evaluate(current_state)
            return self._normalize_eval_score(eval_score)
        
        # Otherwise score based on winner
        winner = current_state.get_winner()
        if winner == "TIGER":
            return 1.0
        elif winner == "GOAT":
            return 0.0
        else:
            return 0.5  # Draw
    
    def _guided_rollout(self, state: GameState) -> float:
        """Perform a rollout guided by the evaluation function."""
        current_state = state.clone()
        max_depth = 30  # Reduced from 50 to limit rollout time
        depth = 0
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        # Debug flag - enable to track evaluation scores during rollout
        debug = False
        
        # Keep track of whether a capture was made in the rollout
        capture_made = False
        
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
            
            if self.use_minimax_eval:
                # Use minimax evaluation to guide rollout while maintaining randomness
                # This approach is common in MCTS implementations
                
                # For both players, evaluate all moves
                move_scores = []
                has_capture_moves = False
                
                for move in valid_moves:
                    # Check for capture moves
                    if move.get("capture"):
                        has_capture_moves = True
                        
                    next_state = current_state.clone()
                    next_state.apply_move(move)
                    
                    # Get raw score from minimax
                    raw_score = self.minimax_agent.evaluate(next_state)
                    
                    # Adjust score based on player perspective
                    if current_state.turn == "GOAT":
                        adjusted_score = -raw_score  # Invert for goat (lower is better)
                    else:
                        adjusted_score = raw_score  # Higher is better for tiger
                        
                    # Add epsilon to ensure all moves have some chance
                    score = max(adjusted_score, -10000) + 10001  # Ensure positive values
                    move_scores.append((move, score))
                
                # More aggressive weighting for high-scoring moves
                # This helps MCTS to better differentiate between good and mediocre moves
                # Square the scores to accentuate differences
                move_scores = [(move, score**1.5) for move, score in move_scores]
                
                # Calculate probabilities - better moves get higher probability
                total_score = sum(score for _, score in move_scores)
                probabilities = [score/total_score for _, score in move_scores]
                
                # Select move based on probabilities (weighted random)
                # This maintains exploration while biasing toward better moves
                moves = [move for move, _ in move_scores]
                selected_move = random.choices(moves, weights=probabilities, k=1)[0]
                
                # Track if a capture was made
                if selected_move.get("capture"):
                    capture_made = True
            else:
                # Fall back to random selection if not using minimax
                selected_move = random.choice(valid_moves)
                
            current_state.apply_move(selected_move)
            depth += 1
        
        # If we hit max depth, use evaluation function
        if depth >= max_depth and self.use_minimax_eval:
            eval_score = self.minimax_agent.evaluate(current_state)
            return self._normalize_eval_score(eval_score)
        
        # Otherwise score based on winner
        winner = current_state.get_winner()
        if winner == "TIGER":
            result = 1.0
        elif winner == "GOAT":
            result = 0.0
        else:
            result = 0.5  # Draw
            
        # Add slight bonus for paths that found captures
        # This helps MCTS recognize the value of positions leading to captures
        if current_state.turn == "TIGER" and capture_made:
            result = min(1.0, result + 0.05)  # Give a small bonus but keep ≤ 1.0
            
        return result

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

    def _normalize_eval_score(self, score: float) -> float:
        """
        Normalize a minimax evaluation score to a value in [0, 1] range.
        1.0 represents Tiger win, 0.0 represents Goat win.
        
        Uses a piecewise linear scaling to better preserve differences in evaluation scores.
        """
        # Debug flag - set to True to log score normalization
        debug = True
        
        # Define thresholds based on domain knowledge of the minimax evaluation
        MIN_SCORE = -5000  # Worst possible score for Tiger 
        NEUTRAL_SCORE = 0  # Neutral position
        GOOD_SCORE = 1000  # Good score for Tiger (1 capture)
        MAX_SCORE = 10000  # Best possible score (game won)
        
        # Clamp score to our defined range
        score = max(min(score, MAX_SCORE), MIN_SCORE)
        
        # Use a more aggressive scaling for scores in different ranges
        # This creates more separation between capture and non-capture moves
        if score >= GOOD_SCORE:
            # Between good and max: 0.7 to 1.0
            # Use a power function to increase separation at higher scores
            ratio = (score - GOOD_SCORE) / (MAX_SCORE - GOOD_SCORE)
            # Apply curve: slower growth at first, faster at higher scores
            ratio = ratio ** 0.7  # Raising to power < 1 creates more separation for high scores
            normalized = 0.7 + 0.3 * ratio
        elif score >= NEUTRAL_SCORE:
            # Between neutral and good: 0.5 to 0.7
            normalized = 0.5 + 0.2 * (score / GOOD_SCORE)
        else:
            # Between min and neutral: 0.0 to 0.5
            normalized = 0.5 * (1.0 + (score / (-MIN_SCORE)))
        
        # Ensure bounds
        normalized = max(0.0, min(1.0, normalized))
        
        if debug:
            print(f"Minimax score: {score}, normalized to: {normalized}")
            # Reference points for interpretation
            good_norm = 0.7
            neutral_norm = 0.5
            print(f"  Score of 1000 (1 goat captured) normalizes to: {good_norm}")
            print(f"  Score of 500 (1 threatened goat) normalizes to: {0.5 + 0.2 * (500 / GOOD_SCORE)}")
            print(f"  Score of 0 normalizes to: {neutral_norm}")
            print(f"  Score of -500 normalizes to: {0.5 * (1.0 + (-500 / (-MIN_SCORE)))}")
            
        return normalized 