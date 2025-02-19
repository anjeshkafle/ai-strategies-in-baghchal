from typing import List, Optional, Dict, Union
from models.game_state import GameState
from game_logic import get_all_possible_moves, check_game_end
import logging
from operator import itemgetter

logger = logging.getLogger(__name__)

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    Implements evaluation and search similar to the reference implementation.
    """
    
    INF = 1000000
    
    def __init__(self, max_depth: int = 5, max_time: Optional[float] = None):
        self.max_depth = max_depth
        self.max_time = max_time  # Not used but kept for compatibility
        self.best_move = None
        self.move_scores = []  # Store scores for all top-level moves
    
    def evaluate(self, state: GameState, depth: int = 0) -> float:
        """
        Evaluates the current game state from Tiger's perspective.
        Positive values favor tigers, negative values favor goats.
        
        Following reference implementation:
        - 300 * movable_tigers
        - 700 * dead_goats
        - -700 * closed_spaces
        - -depth (to prefer faster wins/losses)
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner:
            if winner == "TIGER":
                return MinimaxAgent.INF
            else:
                return -MinimaxAgent.INF
        
        # Core evaluation based on reference implementation
        score = 0
        
        # Count movable tigers (tigers with at least one valid move)
        movable_tigers = self._count_movable_tigers(state)
        tiger_score = 300 * movable_tigers
        score += tiger_score
        
        # Dead goats (captured)
        capture_score = 700 * state.goats_captured
        score += capture_score
        
        # Count closed spaces (positions where tigers are trapped)
        closed_spaces = self._count_closed_spaces(state)
        closed_score = -700 * closed_spaces
        score += closed_score
        
        # Penalize for depth to prefer faster wins/losses
        score -= depth

        # Store evaluation components for logging
        if hasattr(self, 'current_move'):
            self.current_eval = {
                'movable_tigers': movable_tigers,
                'goats_captured': state.goats_captured,
                'closed_spaces': closed_spaces,
                'depth_penalty': depth,
                'total_score': score
            }
        
        return score
    
    def _count_movable_tigers(self, state: GameState) -> int:
        """
        Counts the number of tigers that have at least one valid move.
        This matches the reference implementation's movable_tigers() function.
        """
        movable_count = 0
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                piece = state.board[y][x]
                if piece and piece["type"] == "TIGER":
                    moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
                    tiger_moves = [m for m in moves if m["from"]["x"] == x and m["from"]["y"] == y]
                    if len(tiger_moves) > 0:  # Tiger has at least one move
                        movable_count += 1
        return movable_count
    
    def _count_closed_spaces(self, state: GameState) -> int:
        """
        Counts the number of positions where tigers are trapped.
        A space is considered "closed" if a tiger has no moves.
        This matches the reference implementation's no_of_closed_spaces.
        """
        closed_count = 0
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                piece = state.board[y][x]
                if piece and piece["type"] == "TIGER":
                    moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
                    tiger_moves = [m for m in moves if m["from"]["x"] == x and m["from"]["y"] == y]
                    if len(tiger_moves) == 0:  # Tiger has no moves
                        closed_count += 1
        return closed_count
    
    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        indent = "  " * (self.max_depth - depth)
        
        # Get score at leaf nodes or terminal states
        if depth == 0 or check_game_end(state.board, state.goats_captured) != "PLAYING":
            score = self.evaluate(state, depth)
            logger.info(f"{indent}Leaf node at depth {depth}")
            logger.info(f"{indent}Turn: {state.turn}")
            logger.info(f"{indent}Score: {score}")
            logger.info(f"{indent}Goats captured: {state.goats_captured}")
            logger.info(f"{indent}Evaluation details:")
            logger.info(f"{indent}- Movable tigers: {self._count_movable_tigers(state)} (score: {300 * self._count_movable_tigers(state)})")
            logger.info(f"{indent}- Goats captured: {state.goats_captured} (score: {700 * state.goats_captured})")
            logger.info(f"{indent}- Closed spaces: {self._count_closed_spaces(state)} (score: {-700 * self._count_closed_spaces(state)})")
            logger.info(f"{indent}- Depth penalty: -{depth}")
            return score
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(state, depth)
        
        if not is_maximizing:  # Minimizing player (Goat)
            value = MinimaxAgent.INF
            for move in valid_moves:
                # Log move being considered
                if depth >= self.max_depth - 2:
                    logger.info(f"\n{indent}Depth {depth} (Goat's turn) considering:")
                    if move['type'] == 'placement':
                        logger.info(f"{indent}Place at ({move['x']}, {move['y']})")
                    else:
                        logger.info(f"{indent}Move from ({move['from']['x']}, {move['from']['y']}) to ({move['to']['x']}, {move['to']['y']})")
                
                # Make move
                new_state = state.clone()
                new_state.apply_move(move)
                
                # Store current move for logging at root level
                if depth == self.max_depth:
                    self.current_move = move
                
                # Check if this move allows an immediate capture by Tiger
                tiger_moves = get_all_possible_moves(new_state.board, "MOVEMENT", "TIGER")
                allows_capture = any(m.get('capture') for m in tiger_moves)
                
                # If move allows capture, evaluate immediately
                capture_score = None
                if allows_capture:
                    # Simulate the capture
                    best_capture_score = -MinimaxAgent.INF
                    for tiger_move in tiger_moves:
                        if tiger_move.get('capture'):
                            capture_state = new_state.clone()
                            capture_state.apply_move(tiger_move)
                            score = self.evaluate(capture_state, depth-1)
                            best_capture_score = max(best_capture_score, score)
                    capture_score = best_capture_score
                    
                    if depth >= self.max_depth - 2:
                        logger.info(f"{indent}WARNING: This move allows a capture!")
                        logger.info(f"{indent}Immediate capture score: {capture_score}")
                
                # Get score from child nodes
                value_t = self.minimax(new_state, depth - 1, alpha, beta, True)
                
                # For moves that allow capture, use the worse of the immediate evaluation or child node score
                if capture_score is not None:
                    value_t = max(value_t, capture_score)  # Use max because a higher score is worse for Goat
                
                if depth >= self.max_depth - 2:
                    logger.info(f"{indent}Move got score: {value_t}")
                    if allows_capture:
                        logger.info(f"{indent}Final score after considering capture: {value_t}")
                
                # At root node, always log the move for debugging
                if depth == self.max_depth:
                    self.move_scores.append({
                        'move': move,
                        'score': value_t,
                        'eval': self.current_eval.copy() if hasattr(self, 'current_eval') else None
                    })
                
                # Update best move and value
                if value_t < value:
                    value = value_t
                    beta = min(beta, value)
                    if depth == self.max_depth:
                        self.best_move = move
                        logger.info(f"{indent}New best move for Goat with score: {value_t}")
                
                if alpha >= beta:
                    if depth >= self.max_depth - 2:
                        logger.info(f"{indent}Branch pruned (α={alpha}, β={beta})")
                    break
            
            return value
        else:  # Maximizing player (Tiger)
            value = -MinimaxAgent.INF
            for move in valid_moves:
                # Log move being considered
                if depth >= self.max_depth - 2:
                    logger.info(f"\n{indent}Depth {depth} (Tiger's turn) considering:")
                    if move['type'] == 'placement':
                        logger.info(f"{indent}Place at ({move['x']}, {move['y']})")
                    else:
                        logger.info(f"{indent}Move from ({move['from']['x']}, {move['from']['y']}) to ({move['to']['x']}, {move['to']['y']})")
                        if move.get('capture'):
                            logger.info(f"{indent}This is a capture move!")
                
                # Make move
                new_state = state.clone()
                new_state.apply_move(move)
                
                # Store current move for logging at root level
                if depth == self.max_depth:
                    self.current_move = move
                
                # For capture moves, evaluate immediately and propagate if better
                capture_score = None
                if move.get('capture'):
                    capture_score = self.evaluate(new_state, depth-1)
                    if depth >= self.max_depth - 2:
                        logger.info(f"{indent}State immediately after capture:")
                        logger.info(f"{indent}- Movable tigers: {self._count_movable_tigers(new_state)}")
                        logger.info(f"{indent}- Goats captured: {new_state.goats_captured}")
                        logger.info(f"{indent}- Closed spaces: {self._count_closed_spaces(new_state)}")
                        logger.info(f"{indent}- Raw evaluation: {capture_score}")
                
                # Get score from child nodes
                value_t = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                # For capture moves, use the better of the immediate evaluation or child node score
                if capture_score is not None:
                    value_t = max(value_t, capture_score)
                
                if depth >= self.max_depth - 2:
                    logger.info(f"{indent}Move got score: {value_t}")
                    if move.get('capture'):
                        logger.info(f"{indent}Capture move resulted in score: {value_t}")
                
                # At root node, always log the move for debugging
                if depth == self.max_depth:
                    self.move_scores.append({
                        'move': move,
                        'score': value_t,
                        'eval': self.current_eval.copy() if hasattr(self, 'current_eval') else None
                    })
                
                # Update best move and value
                if value_t > value:
                    value = value_t
                    alpha = max(alpha, value)
                    if depth == self.max_depth:
                        self.best_move = move
                        logger.info(f"{indent}New best move for Tiger with score: {value_t}")
                
                if alpha >= beta:
                    if depth >= self.max_depth - 2:
                        logger.info(f"{indent}Branch pruned (α={alpha}, β={beta})")
                    break
            
            return value
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state."""
        # Reset move scores for new search
        self.move_scores = []
        
        # Log initial state evaluation
        initial_score = self.evaluate(state, 0)
        logger.info("\nINITIAL STATE ANALYSIS:")
        logger.info(f"Movable tigers: {self._count_movable_tigers(state)}")
        logger.info(f"Goats captured: {state.goats_captured}")
        logger.info(f"Trapped tigers: {self._count_closed_spaces(state)}")
        logger.info(f"Initial position score: {initial_score}")
        
        self.best_move = None
        is_maximizing = state.turn == "TIGER"
        
        # Start minimax search
        final_score = self.minimax(
            state,
            self.max_depth,
            -MinimaxAgent.INF,
            MinimaxAgent.INF,
            is_maximizing
        )
        
        # Sort and log all considered moves
        sorted_moves = sorted(self.move_scores, key=itemgetter('score'), reverse=is_maximizing)
        
        logger.info("\nALL CONSIDERED MOVES (sorted by score):")
        for move_data in sorted_moves:
            move = move_data['move']
            score = move_data['score']
            eval_data = move_data['eval']
            
            if move['type'] == 'placement':
                logger.info(f"\nPlace at ({move['x']}, {move['y']}):")
            else:
                logger.info(f"\nMove from ({move['from']['x']}, {move['from']['y']}) to ({move['to']['x']}, {move['to']['y']}):")
                if move.get('capture'):
                    logger.info("This move includes a capture!")
            
            logger.info(f"Score: {score}")
            if eval_data:
                logger.info("Evaluation components:")
                logger.info(f"- Movable tigers: {eval_data['movable_tigers']} (score: {eval_data['movable_tigers'] * 300})")
                logger.info(f"- Goats captured: {eval_data['goats_captured']} (score: {eval_data['goats_captured'] * 700})")
                logger.info(f"- Closed spaces: {eval_data['closed_spaces']} (score: {eval_data['closed_spaces'] * -700})")
                logger.info(f"- Depth penalty: -{eval_data['depth_penalty']}")
        
        logger.info("\nFINAL DECISION:")
        if self.best_move:
            if self.best_move['type'] == "placement":
                logger.info(f"Place at ({self.best_move['x']}, {self.best_move['y']})")
            else:
                logger.info(f"Move from ({self.best_move['from']['x']}, {self.best_move['from']['y']}) to ({self.best_move['to']['x']}, {self.best_move['to']['y']})")
                if self.best_move.get('capture'):
                    logger.info("This move includes a capture!")
            logger.info(f"Expected score after move: {final_score}")
        
        if not self.best_move:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            logger.warning("No best move found, falling back to first valid move")
            return valid_moves[0]
        
        return self.best_move 