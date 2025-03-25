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
        Select a child node using the standard UCB formula.
        """
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def ucb_score(child):
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
            
            # Standard UCB formula
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
                    minimax_score = self.evaluate_move(state, child.move)
                    print(f"   Minimax evaluation: {minimax_score}")
                print("=====================\n")
            
            # Select the best child according to visits (standard MCTS approach)
            if not root.children:
                return None  # No valid moves
                
            # Log all capture moves for analysis
            capture_moves = [(child, child.value / child.visits if child.visits > 0 else 0) 
                            for child in root.children if child.move.get("capture")]
            if capture_moves:
                print("\nAll capture moves:")
                for child, win_rate in capture_moves:
                    print(f"  Capture move: visits={child.visits}, win_rate={win_rate:.2f}")
            
            # Standard MCTS selection: choose child with most visits
            best_child = max(root.children, key=lambda c: c.visits)
            
            # Output details about the selection
            win_rate = best_child.value / best_child.visits if best_child.visits > 0 else 0
            capture_text = "CAPTURE MOVE" if best_child.move.get("capture") else "regular move"
            print(f"\nSelected {capture_text} with {best_child.visits} visits ({best_child.visits/root.visits*100:.1f}% of total)")
            print(f"Win rate: {win_rate:.2f}")
            
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
        """Perform a completely random rollout."""
        import time
        start_time = time.time()
        rollout_timeout = 0.5  # Maximum time for a single rollout in seconds
        
        current_state = state.clone()
        max_depth = self.max_rollout_depth
        depth = 0
        
        # Track visited states to detect repetition
        visited_states = {}  # Format: {state_hash: count}
        
        while not current_state.is_terminal() and depth < max_depth:
            # Check timeout to prevent infinite loops
            if depth % 5 == 0 and time.time() - start_time > rollout_timeout:
                # If timeout, use evaluation function
                eval_score = self.minimax_agent.evaluate(current_state)
                # The evaluation function returns higher values for Tiger advantage
                # We need to normalize to [0,1] where 1 means Tiger win
                return self._normalize_eval_score(eval_score)
            
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
        if depth >= max_depth:
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
        """
        Perform a rollout guided by evaluation function with controllable strictness.
        
        The strictness parameter (0.0 to 1.0) controls how deterministic the rollout is:
        - 0.0: Fully probabilistic selection based on evaluation scores
        - 1.0: Always selects the best evaluated move
        - Values between: Increasingly favor the best moves
        """
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
            
            # Get base evaluation of current state
            base_eval = self.minimax_agent.evaluate(current_state)
            
            # Calculate delta (improvement) for each move
            move_deltas = []
            for move in valid_moves:
                next_state = current_state.clone()
                next_state.apply_move(move)
                
                # Get raw score from minimax
                next_eval = self.minimax_agent.evaluate(next_state)
                
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
                    # For Goat: lower delta is better
                    selected_move = min(move_deltas, key=lambda x: x[1])[0]
            else:
                # Probabilistic selection with strictness-based contrast
                # Higher strictness = more emphasis on better moves
                
                # Calculate power factor based on strictness
                # At strictness=0, power=1 (no change to scores)
                # At strictness=0.9, power=10 (extreme contrast)
                power = 1.0 + (self.guided_strictness * 9.0)  # Ranges from 1.0 to 10.0
                
                # Adjust deltas based on player perspective
                if current_state.turn == "TIGER":
                    # For Tiger: positive deltas are better
                    adjusted_scores = [(move, max(0, delta) + 0.1) for move, delta in move_deltas]
                else:
                    # For Goat: negative deltas are better
                    adjusted_scores = [(move, max(0, -delta) + 0.1) for move, delta in move_deltas]
                
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
        
        # If we hit max depth, use evaluation function
        if depth >= max_depth:
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
        Normalized evaluation scores to a [0,1] range where:
        - 0.0 represents strong goat advantage
        - 1.0 represents strong tiger advantage
        - 0.5 represents a balanced position
        
        Uses a piecewise linear mapping function to provide better discrimination
        between good and bad moves.
        """
        # Clamp to reasonable range
        MAX_SCORE = 3000
        score = max(min(score, MAX_SCORE), -MAX_SCORE)
        
        # Define thresholds based on general observations of the evaluation function
        STRONG_GOAT = 0       # Very strong goat advantage
        MILD_GOAT = 500       # Mild goat advantage
        NEUTRAL = 1000        # Balanced position
        MILD_TIGER = 1500     # Mild tiger advantage
        STRONG_TIGER = 2000   # Strong tiger advantage

        # Apply piecewise linear mapping
        if score <= STRONG_GOAT:
            return 0.05  # Strong goat advantage
        elif score <= MILD_GOAT:
            # Map [STRONG_GOAT, MILD_GOAT] to [0.05, 0.35]
            t = (score - STRONG_GOAT) / (MILD_GOAT - STRONG_GOAT)
            return 0.05 + t * 0.3
        elif score <= NEUTRAL:
            # Map [MILD_GOAT, NEUTRAL] to [0.35, 0.5]
            t = (score - MILD_GOAT) / (NEUTRAL - MILD_GOAT)
            return 0.35 + t * 0.15
        elif score <= MILD_TIGER:
            # Map [NEUTRAL, MILD_TIGER] to [0.5, 0.65]
            t = (score - NEUTRAL) / (MILD_TIGER - NEUTRAL)
            return 0.5 + t * 0.15
        elif score <= STRONG_TIGER:
            # Map [MILD_TIGER, STRONG_TIGER] to [0.65, 0.95]
            t = (score - MILD_TIGER) / (STRONG_TIGER - MILD_TIGER)
            return 0.65 + t * 0.3
        else:
            return 0.95  # Strong tiger advantage 