from typing import List, Optional, Dict, Union
from models.game_state import GameState
from game_logic import get_all_possible_moves
import logging

logger = logging.getLogger(__name__)

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    Implements evaluation and search similar to the reference implementation.
    """
    
    INF = 1000000
    
    def __init__(self, max_depth: int = 5, max_time: Optional[float] = None):  # Added max_time for compatibility
        self.max_depth = max_depth
        self.max_time = max_time  # Not used but kept for compatibility
        self.best_move = None
    
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
        """
        Minimax algorithm with alpha-beta pruning.
        Matches the reference implementation's structure.
        """
        # Get score at leaf nodes
        score = self.evaluate(state, depth)
        
        # Return score if leaf node is reached
        if depth == 0 or abs(score) == MinimaxAgent.INF:
            return score
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return score
        
        if not is_maximizing:  # Minimizing player (Goat)
            value = MinimaxAgent.INF
            for move in valid_moves:
                # Make move
                new_state = state.clone()
                new_state.apply_move(move)
                
                # Recursive evaluation
                value_t = self.minimax(new_state, depth - 1, alpha, beta, True)
                
                if value_t < value:
                    value = value_t
                    beta = min(beta, value)
                    if depth == self.max_depth:  # Root node
                        self.best_move = move
                
                if alpha >= beta:
                    break
            
            return value
        else:  # Maximizing player (Tiger)
            value = -MinimaxAgent.INF
            for move in valid_moves:
                # Make move
                new_state = state.clone()
                new_state.apply_move(move)
                
                # Recursive evaluation
                value_t = self.minimax(new_state, depth - 1, alpha, beta, False)
                
                if value_t > value:
                    value = value_t
                    alpha = max(alpha, value)
                    if depth == self.max_depth:  # Root node
                        self.best_move = move
                
                if alpha >= beta:
                    break
            
            return value
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state."""
        # Log initial state evaluation
        initial_score = self.evaluate(state, 0)
        logger.info("INITIAL STATE ANALYSIS:")
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
        
        # Log final decision
        logger.info("FINAL DECISION:")
        if self.best_move:
            if "type" in self.best_move and self.best_move["type"] == "placement":
                logger.info(f"Place at ({self.best_move['x']}, {self.best_move['y']})")
            else:
                logger.info(f"Move from ({self.best_move['from']['x']}, {self.best_move['from']['y']}) to ({self.best_move['to']['x']}, {self.best_move['to']['y']})")
                if self.best_move.get("capture"):
                    logger.info("This move includes a capture!")
            logger.info(f"Expected score after move: {final_score}")
        
        if not self.best_move:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            logger.warning("No best move found, falling back to first valid move")
            return valid_moves[0]
        
        return self.best_move 