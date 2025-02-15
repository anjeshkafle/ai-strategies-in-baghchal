from typing import List, Optional, Dict, Union
from models.game_state import GameState
from game_logic import get_all_possible_moves

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
        score += 300 * movable_tigers
        
        # Dead goats (captured)
        score += 700 * state.goats_captured
        
        # Count closed spaces (positions where tigers are trapped)
        closed_spaces = self._count_closed_spaces(state)
        score -= 700 * closed_spaces
        
        # Penalize for depth to prefer faster wins/avoid slower losses
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
        """
        Get the best move for the current state.
        
        Args:
            state: Current game state
            
        Returns:
            The best move found
        """
        self.best_move = None
        is_maximizing = state.turn == "TIGER"
        
        # Start minimax search
        self.minimax(
            state,
            self.max_depth,
            -MinimaxAgent.INF,
            MinimaxAgent.INF,
            is_maximizing
        )
        
        if not self.best_move:
            # Fallback to first valid move if something went wrong
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            return valid_moves[0]
        
        return self.best_move 