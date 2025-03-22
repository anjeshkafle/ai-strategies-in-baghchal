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
        """Select a child node using UCT formula."""
        # UCT = value/visits + exploration_weight * sqrt(ln(parent_visits) / visits)
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        def uct_score(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0
            # Add a small epsilon to prevent division by zero
            exploration = exploration_weight * math.sqrt(log_visits / (child.visits + 1e-10)) if child.visits > 0 else float('inf')
            return exploitation + exploration
            
        return max(self.children, key=uct_score)
    
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
            
            # Select the best child of the root based on most visits
            if not root.children:
                return None  # No valid moves
                
            best_child = max(root.children, key=lambda c: c.visits)
            print(f"Selected move with {best_child.visits} visits out of {root.visits} total")
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
                    eval_score = max(min(eval_score, 10000), -10000)
                    normalized_score = 1.0 / (1.0 + math.exp(-eval_score / 1000.0))
                    return normalized_score
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
            # Convert to [0,1] range where 1 is good for Tiger
            # Add clamping to prevent extreme values
            eval_score = max(min(eval_score, 10000), -10000)  # Clamp to reasonable range
            normalized_score = 1.0 / (1.0 + math.exp(-eval_score / 1000.0))
            return normalized_score
        
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
                # Evaluate all moves
                move_scores = []
                for move in valid_moves:
                    score = self.evaluate_move(current_state, move)
                    
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
            # Convert to [0,1] range where 1 is good for Tiger
            # Add clamping to prevent extreme values
            eval_score = max(min(eval_score, 10000), -10000)  # Clamp to reasonable range
            normalized_score = 1.0 / (1.0 + math.exp(-eval_score / 1000.0))
            return normalized_score
        
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
                
            if self.use_minimax_eval:
                # Use evaluation to guide move selection
                if current_state.turn == "TIGER":
                    # For tigers, prefer moves with higher evaluation
                    move_scores = []
                    for move in valid_moves:
                        score = self.evaluate_move(current_state, move)
                        move_scores.append((move, score))
                    
                    move_scores.sort(key=lambda x: x[1], reverse=True)
                    # Select from top 3 moves with some randomness
                    top_moves = move_scores[:min(3, len(move_scores))]
                    selected_move = random.choice([m[0] for m in top_moves])
                else:
                    # For goats, prefer moves with lower evaluation
                    move_scores = []
                    for move in valid_moves:
                        score = self.evaluate_move(current_state, move)
                        move_scores.append((move, score))
                    
                    move_scores.sort(key=lambda x: x[1])
                    # Select from top 3 moves with some randomness
                    top_moves = move_scores[:min(3, len(move_scores))]
                    selected_move = random.choice([m[0] for m in top_moves])
            else:
                # Fall back to random selection
                selected_move = random.choice(valid_moves)
                
            current_state.apply_move(selected_move)
            depth += 1
        
        # If we hit max depth, use evaluation function
        if depth >= max_depth and self.use_minimax_eval:
            eval_score = self.minimax_agent.evaluate(current_state)
            # Convert to [0,1] range where 1 is good for Tiger
            # Add clamping to prevent extreme values
            eval_score = max(min(eval_score, 10000), -10000)  # Clamp to reasonable range
            normalized_score = 1.0 / (1.0 + math.exp(-eval_score / 1000.0))
            return normalized_score
        
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