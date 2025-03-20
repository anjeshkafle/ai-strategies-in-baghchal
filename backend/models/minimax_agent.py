from typing import List, Optional, Dict
from models.game_state import GameState
from game_logic import get_all_possible_moves

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    """
    
    INF = 1000000
    
    def __init__(self, max_depth: int = 5, max_time: Optional[float] = None):
        self.max_depth = max_depth
        self.max_time = max_time  # Not used but kept for compatibility
        self.best_move = None
        self.best_score = None
    
    def evaluate(self, state: GameState, depth: int = 0) -> float:
        """
        Evaluates the current game state from Tiger's perspective.
        Uses five core heuristics:
        - mobility_weight * movable_tigers (200 during placement, 300 during movement)
        - 1000 * dead_goats
        - 500 * threatened_goats
        - -mobility_weight * closed_spaces (200 during placement, 300 during movement)
        - dispersion_weight * tiger_dispersion (100 by default, normalized 0-1 score)
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            final_score = MinimaxAgent.INF - depth  # Prefer faster wins
            return final_score
        elif winner == "GOAT":
            final_score = -MinimaxAgent.INF + depth  # Prefer slower losses from tiger's perspective
            return final_score
        
        # Set mobility weight based on game phase
        mobility_weight = 200 if state.phase == "PLACEMENT" else 300
        
        # Get all tiger moves once for all heuristics
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Core evaluation based on reference implementation
        score = 0
        
        # Count movable tigers (tigers with at least one valid move)
        movable_tigers = self._count_movable_tigers(all_tiger_moves)
        tiger_score = mobility_weight * movable_tigers
        score += tiger_score
        
        # Dead goats (captured)
        capture_score = 1000 * state.goats_captured
        score += capture_score
        
        # Threatened goats (in danger of being captured)
        threatened_value = self._count_threatened_goats(all_tiger_moves)
        threatened_score = 500 * threatened_value
        score += threatened_score
        
        # Count closed spaces (positions where tigers are trapped)
        closed_regions = self._count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        closed_score = -mobility_weight * total_closed_spaces
        score += closed_score
        
        # Calculate tiger dispersion score (normalized 0-1)
        dispersion_score = self._calculate_tiger_dispersion(state)
        dispersion_weight = 100  # Weight for tiger dispersion
        score += dispersion_weight * dispersion_score
        
        # Always subtract depth for non-terminal states
        score -= depth
        
        return score
    
    def _count_movable_tigers(self, all_tiger_moves) -> int:
        """
        Counts the number of tigers that have at least one valid move.
        This matches the reference implementation's movable_tigers() function.
        """
        # Count tigers with at least one move
        movable_tigers = set()
        for move in all_tiger_moves:
            from_pos = (move["from"]["x"], move["from"]["y"])
            movable_tigers.add(from_pos)
        
        return len(movable_tigers)
    
    def _count_closed_spaces(self, state: GameState, all_tiger_moves) -> List[List[tuple[int, int]]]:
        """
        Identifies all closed regions in the current board state.
        A region of connected empty positions is considered "closed" if:
        1. All neighboring positions around the region are occupied by goats
        2. No tiger can access any position in this region through a capture move
        
        Returns:
            List of closed regions, where each region is a list of (x,y) coordinates
            belonging to that region.
        """
        # Filter to only capture moves
        tiger_capture_moves = [move for move in all_tiger_moves if move.get("capture")]
        
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

    def _count_threatened_goats(self, all_tiger_moves) -> float:
        """
        Evaluates the threat value of potential goat captures.
        
        This function uses a non-linear scale that:
        1. Values 1 threatened goat at 1.0
        2. Values 2 threatened goats at 1.9 (close to but less than a capture)
        3. Values 3+ threatened goats at 2.0 (diminishing returns)
        
        Returns:
            A float representing the adjusted threat value.
        """
        # Filter to only capture moves
        capture_moves = [move for move in all_tiger_moves if move.get("capture")]
        total_captures = len(capture_moves)
        
        # Apply the non-linear scale based on number of captures available
        if total_captures == 0:
            return 0
        elif total_captures == 1:
            return 1.0
        elif total_captures == 2:
            # Check if it's the same goat threatened twice
            if capture_moves[0]["capture"]["x"] == capture_moves[1]["capture"]["x"] and \
               capture_moves[0]["capture"]["y"] == capture_moves[1]["capture"]["y"]:
                return 1.5  # Same goat threatened from two directions
            else:
                return 1.9  # Two different goats threatened
        else:
            # For 3+ threats, cap at 2.0
            return 2.0

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
        valid_moves = state.get_valid_moves()
        
        # Order moves based on a shallow evaluation
        ordered_moves = self._order_moves(state, valid_moves)
        
        best_move = None
        best_value = float('-inf') if state.turn == "TIGER" else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            next_state = state.clone()
            next_state.apply_move(move)
            
            next_is_max = next_state.turn == "TIGER"
            value = self.minimax(next_state, self.max_depth - 1, alpha, beta, next_is_max)
            
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
        
        return best_move

    def _order_moves(self, state: GameState, moves: List[Dict]) -> List[Dict]:
        """
        Order moves based on a shallow evaluation.
        For tigers, prioritize captures and threatening moves.
        For goats, prioritize moves that block tiger captures.
        """
        move_scores = []
        
        for move in moves:
            # Apply the move to a cloned state
            next_state = state.clone()
            next_state.apply_move(move)
            
            # Get a quick evaluation score
            score = self.evaluate(next_state)
            
            # For tigers, higher scores are better; for goats, lower scores are better
            if state.turn == "GOAT":
                score = -score  # Invert score for goats to sort in the same direction
                
            # Add a bonus for capture moves to prioritize them
            if move.get("capture"):
                score += 1000  # Large bonus to ensure captures are tried first
                
            move_scores.append((move, score))
        
        # Sort moves by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the ordered moves
        return [move for move, _ in move_scores]

    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool):
        """Minimax algorithm with alpha-beta pruning."""
        # Base cases first
        if depth == 0 or state.is_terminal():
            # Always evaluate from Tiger's perspective
            eval_score = self.evaluate(state, self.max_depth - depth)
            return eval_score
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            eval_score = self.evaluate(state, self.max_depth - depth)
            return eval_score
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(state, valid_moves)
        
        best_value = -MinimaxAgent.INF if is_maximizing else MinimaxAgent.INF
        
        for move in ordered_moves:
            new_state = state.clone()
            new_state.apply_move(move)
            
            # Next turn alternates maximizing/minimizing
            next_is_max = new_state.turn == "TIGER"
            child_score = self.minimax(new_state, depth - 1, alpha, beta, next_is_max)
            
            if is_maximizing:
                best_value = max(best_value, child_score)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, child_score)
                beta = min(beta, best_value)
                
            if beta <= alpha:
                break  # Alpha-beta pruning
        
        return best_value 

    def _calculate_tiger_dispersion(self, state: GameState) -> float:
        """
        Calculates a normalized score (0-1) based on how well-dispersed the tigers are on the board.
        Tigers that are too close together can interfere with each other's capture opportunities.
        
        Returns:
            A normalized score representing how well the tigers are dispersed (higher is better).
            The score is normalized to a range of 0.0 (minimum dispersion) to 1.0 (maximum dispersion).
        """
        # Find all tiger positions
        tiger_positions = []
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if state.board[y][x] is not None and state.board[y][x]["type"] == "TIGER":
                    tiger_positions.append((x, y))
        
        # If we have fewer than 2 tigers, dispersion is not applicable
        if len(tiger_positions) < 2:
            return 0
        
        # Calculate the sum of pairwise distances between tigers
        total_distance = 0
        pairs_count = 0
        
        for i in range(len(tiger_positions)):
            for j in range(i + 1, len(tiger_positions)):
                x1, y1 = tiger_positions[i]
                x2, y2 = tiger_positions[j]
                
                # Use Manhattan distance as it better represents movement on the board
                distance = abs(x1 - x2) + abs(y1 - y2)
                total_distance += distance
                pairs_count += 1
        
        # Normalize by the number of pairs
        avg_distance = total_distance / pairs_count if pairs_count > 0 else 0
        
        # Correct normalization based on actual possible values:
        # For 4 tigers:
        # - Maximum average distance is ~5.33 (32/6) when tigers are at corners
        # - Minimum average distance is ~1.33 (8/6) when tigers are clustered
        
        # Normalize to 0-1 range
        min_avg_distance = 1.33  # Theoretical minimum for 4 tigers
        max_avg_distance = 5.33  # Theoretical maximum for 4 tigers
        
        # Ensure the value stays in range even with fewer tigers
        normalized_score = (avg_distance - min_avg_distance) / (max_avg_distance - min_avg_distance)
        normalized_score = max(0, min(1, normalized_score))  # Clamp to [0,1]
        
        return normalized_score 