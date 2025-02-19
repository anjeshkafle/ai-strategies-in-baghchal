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
        Uses only three core heuristics:
        - 300 * movable_tigers
        - 700 * dead_goats
        - -700 * closed_spaces
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            return MinimaxAgent.INF - depth  # Prefer faster wins
        elif winner == "GOAT":
            return -MinimaxAgent.INF + depth  # Prefer faster losses
        
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
        
        # Store evaluation components for logging
        if hasattr(self, 'current_move'):
            self.current_eval = {
                'movable_tigers': movable_tigers,
                'goats_captured': state.goats_captured,
                'closed_spaces': closed_spaces,
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
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state using minimax with alpha-beta pruning."""
        valid_moves = state.get_valid_moves()
        best_move = None
        best_value = float('-inf') if state.turn == "TIGER" else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        logger.info("\nALL CONSIDERED MOVES:\n")
        
        for move in valid_moves:
            next_state = state.clone()
            next_state.apply_move(move)
            # Tiger maximizes (wants high scores), Goat minimizes (wants low scores)
            value = self.minimax(next_state, self.max_depth - 1, alpha, beta, next_state.turn == "TIGER")
            
            # Log move details
            if move['type'] == 'placement':
                logger.info(f"\nPlace at ({move['x']}, {move['y']}):")
            else:
                logger.info(f"\nMove from ({move['from']['x']}, {move['from']['y']}) to ({move['to']['x']}, {move['to']['y']}):")
                if move.get('capture', False):
                    logger.info("This move includes a capture!")
            logger.info(f"Score: {value}")
            
            # Log evaluation components for this move
            tigers_score = self._count_movable_tigers(next_state) * 300
            goats_captured = next_state.goats_captured * 700
            closed_spaces = self._count_closed_spaces(next_state) * 700
            
            logger.info("Evaluation components:")
            logger.info(f"- Movable tigers: {self._count_movable_tigers(next_state)} (score: {tigers_score})")
            logger.info(f"- Goats captured: {next_state.goats_captured} (score: {goats_captured})")
            logger.info(f"- Closed spaces: {self._count_closed_spaces(next_state)} (score: {-closed_spaces})")
            
            if state.turn == "TIGER":
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:  # GOAT's turn
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
            
        return best_move

    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        # Base cases first
        if depth == 0 or state.is_terminal():
            # Always evaluate from Tiger's perspective
            return self.evaluate(state)
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(state)
        
        # Sort moves to prioritize captures for tigers (helps with alpha-beta pruning)
        if state.turn == "TIGER":
            valid_moves.sort(key=lambda m: 1 if m.get('capture', False) else 0, reverse=True)
        
        value = -MinimaxAgent.INF if is_maximizing else MinimaxAgent.INF
        for move in valid_moves:
            new_state = state.clone()
            new_state.apply_move(move)
            
            # Next turn alternates maximizing/minimizing
            child_score = self.minimax(new_state, depth - 1, alpha, beta, new_state.turn == "TIGER")
            
            if is_maximizing:
                value = max(value, child_score)
                alpha = max(alpha, value)
            else:
                value = min(value, child_score)
                beta = min(beta, value)
                
            if beta <= alpha:
                break
                
        return value 