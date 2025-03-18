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
        self.best_score = None  # Add storage for best score
        self.move_scores = []  # Store scores for all top-level moves
    
    def _order_moves(self, moves: List[Dict], player: str) -> List[Dict]:
        """
        Order moves based on player strategy to improve alpha-beta pruning.
        
        Args:
            moves: List of valid moves
            player: "TIGER" or "GOAT"
            
        Returns:
            Ordered list of moves
        """
        if player == "TIGER":
            # Tigers prioritize capture moves
            return sorted(moves, key=lambda m: 1 if m.get('capture', False) else 0, reverse=True)
        elif player == "GOAT":
            # Goats prioritize moves that don't lead to captures
            if moves and moves[0].get("type") == "movement":
                # For movement phase, analyze which moves might lead to captures
                def might_lead_to_capture(move):
                    # Create a new board state with this move applied
                    from models.game_state import GameState  # Import here to avoid circular imports
                    
                    # We need the original state to clone it
                    if hasattr(self, 'current_state'):
                        state = self.current_state.clone()
                        state.apply_move(move)
                        state.turn = "TIGER"  # Switch to tiger's turn
                        
                        # Get all possible tiger moves
                        tiger_moves = state.get_valid_moves()
                        
                        # Check if any tiger move is a capture
                        return any("capture" in m for m in tiger_moves)
                    
                    # If we don't have the current state, we can't analyze
                    # Just use a default ordering
                    return False
                
                # Try to sort by whether moves might lead to captures
                # This is more expensive but more accurate
                try:
                    return sorted(moves, key=might_lead_to_capture)
                except:
                    # If analysis fails, fall back to default ordering
                    return moves
        return moves
    
    def evaluate(self, state: GameState, depth: int = 0) -> float:
        """
        Evaluates the current game state from Tiger's perspective.
        Uses only three core heuristics:
        - 300 * movable_tigers
        - 700 * dead_goats
        - -700 * closed_spaces
        
        Note: The closed_spaces calculation currently just counts total spaces,
        but the _count_closed_spaces method now returns detailed region information
        that could be used for more sophisticated evaluation in the future
        (e.g., weighting larger regions differently, or considering region shapes).
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            final_score = MinimaxAgent.INF - depth  # Prefer faster wins
            return final_score
        elif winner == "GOAT":
            final_score = -MinimaxAgent.INF + depth  # Prefer slower losses from tiger's perspective
            return final_score
        
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
        closed_regions = self._count_closed_spaces(state)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        closed_score = -700 * total_closed_spaces
        score += closed_score
        
        # Store evaluation components for logging
        if hasattr(self, 'current_move'):
            self.current_eval = {
                'movable_tigers': movable_tigers,
                'goats_captured': state.goats_captured,
                'closed_spaces': total_closed_spaces,
                'closed_regions': [len(region) for region in closed_regions],  # Add region sizes for logging
                'total_score': score
            }
        
        # Always subtract depth for non-terminal states
        # This reflects that Tiger always wants to reach its best outcome as quickly as possible
        # regardless of whether the evaluation is positive or negative
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
    
    def _count_closed_spaces(self, state: GameState) -> List[List[tuple[int, int]]]:
        """
        Identifies all closed regions in the current board state.
        A region of connected empty positions is considered "closed" if:
        1. All neighboring positions around the region are occupied by goats
        2. No tiger can access any position in this region through a capture move
        
        Returns:
            List of closed regions, where each region is a list of (x,y) coordinates
            belonging to that region. This allows for more sophisticated evaluation
            of closed spaces in the future (e.g., weighting larger regions differently).
        """
        # Get all tiger capture moves first for efficiency
        tiger_capture_moves = []
        for ty in range(GameState.BOARD_SIZE):
            for tx in range(GameState.BOARD_SIZE):
                piece = state.board[ty][tx]
                if piece and piece["type"] == "TIGER":
                    # Get all possible moves for this tiger
                    moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
                    # Filter to only capture moves
                    capture_moves = [move for move in moves if move.get("capture")]
                    tiger_capture_moves.extend(capture_moves)
        
        # Set of destinations that tigers can capture to
        capturable_positions = {(move["to"]["x"], move["to"]["y"]) for move in tiger_capture_moves}
        
        # Get all empty positions
        empty_positions = []
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if state.board[y][x] is None:
                    empty_positions.append((x, y))
        
        # Track visited positions to avoid reprocessing
        visited = set()
        closed_regions = []
        
        # For each empty position, find its connected region
        for x, y in empty_positions:
            if (x, y) in visited:
                continue
                
            # Find the connected region of empty spaces
            region = []
            is_closed_region = True
            
            # Use BFS to find all connected empty spaces
            queue = [(x, y)]
            region_visited = {(x, y)}
            
            while queue:
                curr_x, curr_y = queue.pop(0)
                region.append((curr_x, curr_y))
                
                # Get neighboring positions
                neighbors = []
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    new_x, new_y = curr_x + dx, curr_y + dy
                    if (0 <= new_x < GameState.BOARD_SIZE and 
                        0 <= new_y < GameState.BOARD_SIZE and 
                        self._is_valid_connection(curr_x, curr_y, new_x, new_y)):
                        neighbors.append((new_x, new_y))
                
                # Check neighbors: if empty, add to queue, if not goat, region not closed
                for nx, ny in neighbors:
                    piece = state.board[ny][nx]
                    
                    if piece is None:
                        # Connected empty space
                        if (nx, ny) not in region_visited:
                            queue.append((nx, ny))
                            region_visited.add((nx, ny))
                    elif piece["type"] != "GOAT":
                        # If any neighbor is not a goat, region is not closed
                        is_closed_region = False
            
            # Mark all positions in this region as visited
            visited.update(region_visited)
            
            # Check if any position in the region can be captured to
            if is_closed_region:
                for pos_x, pos_y in region:
                    if (pos_x, pos_y) in capturable_positions:
                        is_closed_region = False
                        break
            
            # If the region is closed, add it to our list of closed regions
            if is_closed_region:
                closed_regions.append(region)
        
        return closed_regions

    def _is_valid_connection(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Helper method to check if two positions are validly connected on the board."""
        # Orthogonal moves are always valid if adjacent
        if abs(from_x - to_x) + abs(from_y - to_y) == 1:
            return True

        # Diagonal moves need special handling
        if abs(from_x - to_x) == 1 and abs(from_y - to_y) == 1:
            # No diagonal moves for second and fourth nodes on outer edges
            if self._is_outer_layer(from_x, from_y):
                is_second_or_fourth_node = (
                    ((from_x == 0 or from_x == 4) and (from_y == 1 or from_y == 3)) or
                    ((from_y == 0 or from_y == 4) and (from_x == 1 or from_x == 3))
                )
                if is_second_or_fourth_node:
                    return False

            # No diagonal moves for middle nodes in second layer
            if self._is_second_layer(from_x, from_y):
                is_middle_node = (
                    (from_x == 1 and from_y == 2) or
                    (from_x == 2 and from_y == 1) or
                    (from_x == 2 and from_y == 3) or
                    (from_x == 3 and from_y == 2)
                )
                if is_middle_node:
                    return False
            return True
        return False

    def _is_outer_layer(self, x: int, y: int) -> bool:
        """Check if a position is on the outer layer of the board."""
        return x == 0 or y == 0 or x == 4 or y == 4

    def _is_second_layer(self, x: int, y: int) -> bool:
        """Check if a position is on the second layer of the board."""
        return x == 1 or y == 1 or x == 3 or y == 3
    
    def get_move(self, state: GameState) -> Dict:
        """Get the best move for the current state using minimax with alpha-beta pruning."""
        # Store the current state for move ordering analysis
        self.current_state = state
        
        valid_moves = state.get_valid_moves()
        
        # Order moves to improve alpha-beta pruning efficiency
        valid_moves = self._order_moves(valid_moves, state.turn)
        
        best_move = None
        best_value = float('-inf') if state.turn == "TIGER" else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            next_state = state.clone()
            next_state.apply_move(move)
            # Tiger maximizes (wants high scores), Goat minimizes (wants low scores)
            value = self.minimax(next_state, self.max_depth - 1, alpha, beta, next_state.turn == "TIGER")
            
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
        
        # Store the best score for later retrieval
        self.best_score = best_value
        
        # Clean up
        if hasattr(self, 'current_state'):
            delattr(self, 'current_state')
        
        return best_move

    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        # Store the current state for move ordering analysis
        self.current_state = state
        
        # Base cases first
        if depth == 0 or state.is_terminal():
            # Always evaluate from Tiger's perspective
            eval_score = self.evaluate(state, self.max_depth - depth)
            return eval_score
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            eval_score = self.evaluate(state, self.max_depth - depth)
            return eval_score
        
        # Order moves to improve alpha-beta pruning efficiency
        valid_moves = self._order_moves(valid_moves, state.turn)
        
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