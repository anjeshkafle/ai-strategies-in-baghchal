from typing import List, Optional, Dict, Tuple
from models.game_state import GameState
from game_logic import get_all_possible_moves
import sys

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    """
    
    INF = 1000000
    
    def __init__(self, max_depth: int = 5, max_time: Optional[float] = None, randomize_equal_moves: bool = False):
        self.max_depth = max_depth
        self.max_time = max_time  # Not used but kept for compatibility
        self.best_move = None
        self.best_score = None
        self.randomize_equal_moves = randomize_equal_moves  # Flag to control move randomization
        
        # Define all evaluation weights in one place for easy tuning
        # Mobility and space control
        self.mobility_weight_placement = 200     # Weight for movable tigers during placement
        self.mobility_weight_movement = 300      # Weight for movable tigers during movement
        self.closed_spaces_weight = 1000         # Weight for closed spaces (always 1000)
        
        # Capture-related weights
        self.base_capture_value = 3000           # Base value for each captured goat
        self.capture_speed_weight = 20000          # Weight for depth-sensitive capture bonus
        
        # Positioning weights
        self.dispersion_weight = 100             # Weight for tiger dispersion
        self.edge_weight = 300                   # Weight for goat edge preference
        
        # Debug mode flag
        self.debug_mode = True
    
    def evaluate(self, state: GameState, move_sequence: List[Dict] = None) -> float:
        """
        Evaluates the current game state from Tiger's perspective using dynamic equilibrium points.
        Uses several core heuristics with weights and balance points that adapt to game progression:
        - Captures and threatened goats
        - Movable tigers and closed spaces
        - Tiger position, optimal spacing, and goat edge preference with dynamic equilibrium points
        """
        # Initialize move_sequence if None
        if move_sequence is None:
            move_sequence = []
            
        # Get move sequence length
        move_sequence_length = len(move_sequence)
        
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            final_score = MinimaxAgent.INF - move_sequence_length  # Prefer faster wins
            return final_score
        elif winner == "GOAT":
            final_score = -MinimaxAgent.INF + move_sequence_length  # Prefer slower losses from tiger's perspective
            return final_score
        
        # Compute the raw score based on board state and phase with dynamic equilibrium
        raw_score = self._compute_raw_score(state)
        
        # Adjust the score based on move sequence and captures
        final_score = self._adjust_score(raw_score, state, move_sequence)
        
        return final_score
     
    def _compute_raw_score(self, state: GameState) -> float:
        """
        Computes the raw evaluation score with dynamic equilibrium points and weights that adapt to game progression.
        This enhances the evaluation function by recognizing how the importance of different factors changes throughout
        the game, similar to the MCTS win rate predictor's approach.
        """
        # Set mobility weight based on game phase
        mobility_weight = self.mobility_weight_placement if state.phase == "PLACEMENT" else self.mobility_weight_movement
        
        # Track game progression
        goats_placed = state.goats_placed
        
        # Get all tiger moves once for all heuristics
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Initialize score
        score = 0
        
        # Count movable tigers (tigers with at least one valid move)
        movable_tigers = self._count_movable_tigers(all_tiger_moves)
        tiger_score = mobility_weight * movable_tigers
        score += tiger_score
        
        # Threatened goats (in danger of being captured)
        threatened_value = self._count_threatened_goats(all_tiger_moves, state.turn)
        threatened_score = self.base_capture_value * threatened_value
        score += threatened_score
        
        # Calculate stage dependent weights based on goats_placed
        early_game_ratio = max(0, 1 - (goats_placed / 15))  # Decreases from 1.0 to 0.0 as goats_placed goes from 0 to 15
        late_game_ratio = min(1, goats_placed / 15)         # Increases from 0.0 to 1.0 as goats_placed goes from 0 to 15
        
        # Count closed spaces (positions where tigers are trapped)
        closed_regions = self._count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Make closed spaces weight dynamic - more important in late game
        # Closed spaces become increasingly important as more goats are placed
        closed_space_weight_factor = 1.0 + (1.5 * late_game_ratio)  # Ranges from 1.0 to 2.5
        dynamic_closed_space_weight = self.closed_spaces_weight * closed_space_weight_factor
        
        closed_score = -dynamic_closed_space_weight * total_closed_spaces
        score += closed_score
        
        # Calculate tiger positional score (normalized 0-1)
        position_score = self._calculate_tiger_positional_score(state)
        
        # Calculate tiger optimal spacing score (normalized 0-1)
        optimal_spacing_score = self._calculate_tiger_optimal_spacing(state)
        
        # Calculate goat edge preference score (normalized 0-1)
        edge_score = self._calculate_goat_edge_preference(state)
        
        # Calculate dynamic equilibrium points for each heuristic based on game stage
        
        # Tiger position score equilibrium: 0.5 before 10 goats placed, decreases to 0.33 by 15 goats placed
        if goats_placed < 10:
            position_equilibrium = 0.5
        elif goats_placed < 15:
            # Linear interpolation from 0.5 to 0.33 between 10 and 15 goats placed
            position_equilibrium = 0.5 - (0.17 * (goats_placed - 10) / 5)
        else:
            position_equilibrium = 0.33
            
        # Tiger optimal spacing equilibrium: 0.5 before 10 goats placed, decreases to 0.33 by 15 goats placed
        if goats_placed < 10:
            spacing_equilibrium = 0.5
        elif goats_placed < 15:
            # Linear interpolation from 0.5 to 0.33 between 10 and 15 goats placed
            spacing_equilibrium = 0.5 - (0.17 * (goats_placed - 10) / 5)
        else:
            spacing_equilibrium = 0.33
            
        # Goat edge preference equilibrium: 1.0 before 5 goats placed, 0.8 until 12 goats placed, decreases to 0.1 by 20 goats placed
        if goats_placed < 5:
            edge_equilibrium = 1.0
        elif goats_placed < 12:
            edge_equilibrium = 0.8
        elif goats_placed <= 20:
            # Linear interpolation from 0.8 to 0.1 between 12 and 20 goats placed
            edge_equilibrium = 0.8 - (0.7 * (goats_placed - 12) / 8)
        else:
            edge_equilibrium = 0.1
        
        # Calculate factor values based on equilibrium points
        position_factor = position_score - position_equilibrium
        spacing_factor = optimal_spacing_score - spacing_equilibrium
        edge_factor = edge_equilibrium - edge_score  # Note the inverted formula because higher edge score favors goats
        
        # Position weight is more important in early game
        position_weight_factor = 1.0 + (0.5 * early_game_ratio)  # Ranges from 1.0 to 1.5
        # Edge weight is more important in early game
        edge_weight_factor = 1.0 + (1.0 * early_game_ratio)      # Ranges from 1.0 to 2.0
        # Spacing weight increases in mid-to-late game
        spacing_weight_factor = 1.0 + (0.7 * late_game_ratio)    # Ranges from 1.0 to 1.7
        
        # Apply dynamic position weight
        position_weight = self.dispersion_weight * position_weight_factor
        score += position_weight * position_factor
        
        # Apply dynamic spacing weight
        optimal_spacing_weight = int(self.dispersion_weight * 1.5 * spacing_weight_factor)
        score += optimal_spacing_weight * spacing_factor
        
        # Apply dynamic edge weight 
        dynamic_edge_weight = self.edge_weight * edge_weight_factor
        score += dynamic_edge_weight * edge_factor  # Note: Using positive now since we flipped the factor calculation
        
        return score
    
   
    def _adjust_score(self, raw_score: float, state: GameState, move_sequence: List[Dict]) -> float:
        """
        Adjusts the raw evaluation score by applying move sequence length penalty and capture bonuses.
        Uses move_sequence to determine captures.
        """
        # Get move sequence length
        move_sequence_length = len(move_sequence)
        
        # Start with the raw score
        adjusted_score = raw_score
        
        # Calculate game stage for dynamic capture value
        game_progress = min(1.0, state.goats_placed / 20)
        
        # Calculate dynamic capture value - captures are more valuable earlier in the game
        early_game_bonus = max(0, 1.2 - game_progress)  # Bonus ranges from 1.2 to 0.0
        dynamic_capture_value = self.base_capture_value * (1 + early_game_bonus)
        
        # Add regular capture score for goats that have already been captured
        base_capture_score = dynamic_capture_value * state.goats_captured
        adjusted_score += base_capture_score
        
        # Add bonuses for captures in the move sequence
        for i, move in enumerate(move_sequence):
            if move.get("capture"):
                # Calculate bonus based on how early in the sequence the capture occurs
                remaining_depth = self.max_depth - i
                capture_bonus = self.capture_speed_weight * remaining_depth
                adjusted_score += capture_bonus
        # Apply move sequence length penalty
        adjusted_score -= move_sequence_length
        
        return adjusted_score
    
    
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
        2. None of the goats forming the "wall" around the region are threatened
        
        Returns:
            List of closed regions, where each region is a list of (x,y) coordinates
            belonging to that region.
        """
        # Filter to only capture moves
        tiger_capture_moves = [move for move in all_tiger_moves if move.get("capture")]
        
        # Set of goats that can be captured (positions of threatened goats)
        capturable_goats = {(move["capture"]["x"], move["capture"]["y"]) for move in tiger_capture_moves}
        
        # Track visited positions to avoid reprocessing
        visited = set()
        closed_regions = []
        
        # Directly iterate through the board instead of creating an empty_positions list
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                # Skip if already visited or not empty
                if (x, y) in visited or state.board[y][x] is not None:
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
                    
                    # Check neighbors: if empty, add to queue; if goat, check if capturable; if tiger, region not closed
                    for nx, ny in neighbors:
                        piece = state.board[ny][nx]
                        
                        if piece is None:
                            # Connected empty space
                            if (nx, ny) not in region_visited:
                                queue.append((nx, ny))
                                region_visited.add((nx, ny))
                        elif piece["type"] == "GOAT":
                            # If this goat can be captured, the region is not closed
                            if (nx, ny) in capturable_goats:
                                is_closed_region = False
                        else:
                            # If any neighbor is a tiger, region is not closed
                            is_closed_region = False
                
                # Mark all positions in this region as visited
                visited.update(region_visited)
                
                # If the region is closed, add it to our list of closed regions
                if is_closed_region:
                    closed_regions.append(region)
        
        return closed_regions

    def _count_threatened_goats(self, all_tiger_moves, turn) -> float:
        """
        Evaluates the threat value of potential goat captures, taking into account whose turn it is.
        
        If it's Tiger's turn to play next:
          - Return 1.0 for any number of threats (equivalent to a capture)
        
        If it's Goat's turn to play next:
          - 1 threat: 0.3 (goat can likely escape)
          - 2+ threats: 0.5 (harder to defend but still possible)
        
        Returns:
            A float representing the adjusted threat value.
        """
        # Filter to only capture moves
        capture_moves = [move for move in all_tiger_moves if move.get("capture")]
        total_captures = len(capture_moves)
        
        if total_captures == 0:
            return 0
            
        # Turn-dependent evaluation
        if turn == "TIGER":
            # If tiger can capture immediately, value this as highly as a capture
            return 1.0
        else:  # GOAT's turn
            if total_captures == 1:
                return 0.3  # One threat, goat can likely escape
            else:
                return 0.5  # Multiple threats are harder to defend

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
        
        best_moves = []  # List to collect equally good moves
        best_value = float('-inf') if state.turn == "TIGER" else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Debug: Track move evaluations
        move_evals = []
        
        for move in ordered_moves:
            next_state = state.clone()
            next_state.apply_move(move)
            
            next_is_max = next_state.turn == "TIGER"
            # Create move sequence with this first move and pass it to minimax
            move_sequence = [move]
            
            # Get the score
            value = self.minimax(next_state, move_sequence, alpha, beta, next_is_max)
            
            # Save evaluation for debugging
            if self.debug_mode:
                move_evals.append((move, value))
            
            if state.turn == "TIGER":
                if value > best_value:
                    best_value = value
                    best_moves = [move]  # Reset the list with this new best move
                elif value == best_value:
                    best_moves.append(move)  # Add to list of equally good moves
                alpha = max(alpha, value)
            else:  # GOAT's turn
                if value < best_value:
                    best_value = value
                    best_moves = [move]  # Reset the list with this new best move
                elif value == best_value:
                    best_moves.append(move)  # Add to list of equally good moves
                beta = min(beta, value)
        
        # Debug: Print move evaluations and best moves
        if self.debug_mode:
            print("Move evaluations:")
            for move, score in move_evals:
                print(f"  {move}: {score}")
            
            print(f"\nBest moves: {best_moves}")
            print(f"Best value: {best_value}")
            print(f"Randomizing: {self.randomize_equal_moves}")
        
        # Store the best score for later retrieval
        self.best_score = best_value
        
        # Select a random move from the best moves if randomization is enabled and we have multiple options
        if self.randomize_equal_moves and len(best_moves) > 1:
            import random
            best_move = random.choice(best_moves)
        else:
            # Just take the first (highest scored according to move ordering) move
            best_move = best_moves[0]
        
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
            
            # Get a quick evaluation score - pass empty move sequence
            score = self.evaluate(next_state, [])
            
            # For tigers, higher scores are better; for goats, lower scores are better
            if state.turn == "GOAT":
                score = -score  # Invert score for goats to sort in the same direction
                
            # Add a bonus for capture moves to prioritize them in move ordering
            # This is just for efficiency in alpha-beta pruning
            if move.get("capture"):
                # Use the same capture_speed_weight for move ordering as well
                score += self.capture_speed_weight
                
            move_scores.append((move, score))
        
        # Sort moves by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the ordered moves
        return [move for move, _ in move_scores]

    def _transform_coordinates(self, x: int, y: int, symmetry_type: str) -> Tuple[int, int]:
        """Transform coordinates according to the given symmetry type."""
        if symmetry_type == "identity":
            return x, y
        elif symmetry_type == "rotation_90":
            return y, 4-x
        elif symmetry_type == "rotation_180":
            return 4-x, 4-y
        elif symmetry_type == "rotation_270":
            return 4-y, x
        elif symmetry_type == "flip_horizontal":
            return x, 4-y
        elif symmetry_type == "flip_vertical":
            return 4-x, y
        elif symmetry_type == "flip_diagonal":
            return y, x
        elif symmetry_type == "flip_antidiagonal":
            return 4-y, 4-x
        else:
            raise ValueError(f"Unknown symmetry type: {symmetry_type}")

    def _transform_move(self, move: Dict, symmetry_type: str) -> Dict:
        """Transform a move according to the given symmetry type."""
        if symmetry_type == "identity":
            return move
        
        new_move = move.copy()
        
        # Transform 'from' coordinates if they exist
        if "from" in move:
            from_x, from_y = self._transform_coordinates(
                move["from"]["x"], 
                move["from"]["y"], 
                symmetry_type
            )
            new_move["from"] = {"x": from_x, "y": from_y}
        
        # Transform 'to' coordinates if they exist
        if "to" in move:
            to_x, to_y = self._transform_coordinates(
                move["to"]["x"], 
                move["to"]["y"], 
                symmetry_type
            )
            new_move["to"] = {"x": to_x, "y": to_y}
        
        # Transform 'position' coordinates if they exist (for placement moves)
        if "position" in move:
            pos_x, pos_y = self._transform_coordinates(
                move["position"]["x"], 
                move["position"]["y"], 
                symmetry_type
            )
            new_move["position"] = {"x": pos_x, "y": pos_y}
        
        # Transform 'capture' coordinates if they exist
        if "capture" in move:
            cap_x, cap_y = self._transform_coordinates(
                move["capture"]["x"], 
                move["capture"]["y"], 
                symmetry_type
            )
            new_move["capture"] = {"x": cap_x, "y": cap_y}
        
        return new_move

    def _transform_move_sequence(self, move_sequence: List[Dict], symmetry_type: str) -> List[Dict]:
        """Transform a sequence of moves according to the given symmetry type."""
        return [self._transform_move(move, symmetry_type) for move in move_sequence]

    def minimax(self, state: GameState, move_sequence: List[Dict], alpha: float, beta: float, is_maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        Uses move_sequence to track the sequence of moves leading to the current state.
        
        Returns:
            float: The evaluation score for the position
        """
        # Calculate depth based on move sequence length
        depth = self.max_depth - len(move_sequence)
        
        # Base cases
        if depth == 0 or state.is_terminal():
            # Always evaluate from Tiger's perspective
            return self.evaluate(state, move_sequence)
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(state, move_sequence)
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(state, valid_moves)
        
        best_value = -MinimaxAgent.INF if is_maximizing else MinimaxAgent.INF
        
        for move in ordered_moves:
            new_state = state.clone()
            new_state.apply_move(move)
            
            # Modify move sequence in place and restore after recursion
            move_sequence.append(move)
            
            # Next turn alternates maximizing/minimizing
            next_is_max = new_state.turn == "TIGER"
            child_score = self.minimax(new_state, move_sequence, alpha, beta, next_is_max)
            
            # Restore move sequence
            move_sequence.pop()
            
            if is_maximizing:
                if child_score > best_value:
                    best_value = child_score
                alpha = max(alpha, best_value)
            else:
                if child_score < best_value:
                    best_value = child_score
                beta = min(beta, best_value)
                
            if beta <= alpha:
                break  # Alpha-beta pruning
        
        return best_value
    
    def _calculate_tiger_positional_score(self, state: GameState) -> float:
        """
        Calculates a normalized score (0-1) based on how well tigers are positioned on the board
        using a matrix of board weights that corresponds to the sum of legal moves and captures
        available from each position.
        
        Returns:
            A normalized score representing how well tigers are positioned (higher is better).
            The score is normalized to a range of 0.0 (worst positioning) to 1.0 (best positioning).
        """
        # Board weights matrix - each value represents the value of a position on the board
        board_weights = [
            [6, 5, 10, 5, 6],
            [5, 11, 7, 11, 5],
            [10, 7, 5, 7, 10],
            [5, 11, 7, 11, 5],
            [6, 5, 10, 5, 6]
        ]
        
        # Maximum possible total position score (if all 4 tigers are on the highest valued positions)
        # Take the 4 highest values from the board_weights matrix
        flattened_weights = [weight for row in board_weights for weight in row]
        flattened_weights.sort(reverse=True)
        max_position_score = sum(flattened_weights[:4])  # Sum of the 4 highest position values
        
        # Minimum possible total position score (if all 4 tigers are on the lowest valued positions)
        flattened_weights.sort()
        min_position_score = sum(flattened_weights[:4])  # Sum of the 4 lowest position values
        
        # Find all tiger positions and calculate their position score
        total_position_score = 0
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if state.board[y][x] is not None and state.board[y][x]["type"] == "TIGER":
                    total_position_score += board_weights[y][x]
        
        # Normalize to 0-1 range
        score_range = max_position_score - min_position_score
        if score_range > 0:
            normalized_score = (total_position_score - min_position_score) / score_range
        else:
            normalized_score = 0
        
        return normalized_score
    
    def _calculate_tiger_optimal_spacing(self, state: GameState) -> float:
        """
        Calculates a normalized score (0-1) based on how many tiger pairs are optimally spaced
        with exactly 3 nodes apart (2 empty nodes between them) respecting board connectivity rules.
        
        This is strategically important as goats can't place on either of those empty nodes without
        getting captured.
        
        Returns:
            A normalized score representing optimal tiger spacing (higher is better).
            The score is normalized to a range of 0.0 (no optimal spacing) to 1.0 (maximum optimal spacing).
        """
        # Find all tiger positions
        tiger_positions = []
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if state.board[y][x] is not None and state.board[y][x]["type"] == "TIGER":
                    tiger_positions.append((x, y))
        
        # If we have fewer than 2 tigers, optimal spacing is not applicable
        if len(tiger_positions) < 2:
            return 0
        
        # Count pairs of tigers that are exactly 3 nodes apart
        # (i.e., have exactly 2 empty nodes between them)
        optimal_pairs = 0
        total_pairs = 0
        
        # Check all tiger pairs
        for i in range(len(tiger_positions)):
            for j in range(i + 1, len(tiger_positions)):
                x1, y1 = tiger_positions[i]
                x2, y2 = tiger_positions[j]
                total_pairs += 1
                
                # Check for optimal spacing
                # To be exactly 3 nodes apart (2 empty nodes between), tigers must be in the same row, same column,
                # or along a valid diagonal with 2 empty nodes between them
                
                # Same row
                if y1 == y2 and abs(x1 - x2) == 3:
                    # Check if both intermediary nodes are empty
                    middle_x1 = min(x1, x2) + 1
                    middle_x2 = min(x1, x2) + 2
                    
                    if (state.board[y1][middle_x1] is None and 
                        state.board[y1][middle_x2] is None and
                        self._is_valid_connection(x1, y1, middle_x1, y1)):
                        optimal_pairs += 1
                        continue
                
                # Same column
                if x1 == x2 and abs(y1 - y2) == 3:
                    # Check if both intermediary nodes are empty
                    middle_y1 = min(y1, y2) + 1
                    middle_y2 = min(y1, y2) + 2
                    
                    if (state.board[middle_y1][x1] is None and 
                        state.board[middle_y2][x1] is None and
                        self._is_valid_connection(x1, y1, x1, middle_y1)):
                        optimal_pairs += 1
                        continue
                
                # Check for valid diagonal spacing
                # For tigers to be 3 nodes apart diagonally, the distance must be 3,3
                if abs(x1 - x2) == 3 and abs(y1 - y2) == 3:
                    # Calculate the two intermediary positions
                    dx = 1 if x2 > x1 else -1
                    dy = 1 if y2 > y1 else -1
                    
                    middle_x1, middle_y1 = x1 + dx, y1 + dy
                    middle_x2, middle_y2 = x1 + 2*dx, y1 + 2*dy
                    
                    # Check if both intermediary positions are empty
                    if (state.board[middle_y1][middle_x1] is None and 
                        state.board[middle_y2][middle_x2] is None):
                        
                        # Only need to check first connection to ensure valid diagonal path exists
                        if self._is_valid_connection(x1, y1, middle_x1, middle_y1):
                            optimal_pairs += 1
                            continue
        
        # Normalize based on total possible pairs
        # For 4 tigers, there are 6 possible pairs (4 choose 2)
        # maximum theoretical optimal pairs is 6 if all tigers are optimally spaced
        max_optimal_pairs = total_pairs
        
        if max_optimal_pairs > 0:
            normalized_score = optimal_pairs / max_optimal_pairs
        else:
            normalized_score = 0
            
        return normalized_score
        
    def _calculate_goat_edge_preference(self, state: GameState) -> float:
        """
        Calculates a normalized score (0-1) based on how well goats are positioned on the edges.
        The score is perfect (1.0) when all goats are optimally placed (outer layer preferred,
        then middle layer, with center being worst).
        
        Returns:
            A normalized score representing how well goats are positioned on edges (higher is better for goats).
            The score is normalized to a range of 0.0 (worst positioning) to 1.0 (best positioning).
        """
        # Count goats in each layer
        outer_layer_goats = 0
        middle_layer_goats = 0
        center_goats = 0
        
        for y in range(GameState.BOARD_SIZE):
            for x in range(GameState.BOARD_SIZE):
                if state.board[y][x] is not None and state.board[y][x]["type"] == "GOAT":
                    if self._is_outer_layer(x, y):
                        outer_layer_goats += 1
                    elif self._is_second_layer(x, y):
                        middle_layer_goats += 1
                    else:  # Center position
                        center_goats += 1
        
        # Total number of goats on the board
        total_goats = outer_layer_goats + middle_layer_goats + center_goats
        
        # If no goats on the board, return 1.0 (conceptually all 0 goats are optimally placed)
        if total_goats == 0:
            return 1.0
            
        # Calculate the actual score based on placement quality
        # Perfect score (1.0) if all goats are on the outer layer
        # Reduced score for goats on middle layer (0.67) or center (0.33)
        placement_quality = (outer_layer_goats + (middle_layer_goats * 0.67) + (center_goats * 0.33)) / total_goats
        
        return placement_quality

    def _test_transformations(self, board):
        """Test that all board transformations work correctly."""
        print("\nOriginal board:")
        self._print_board(board)
        
        print("\n90° rotation:")
        rotated90 = self._rotate_90(board)
        self._print_board(rotated90)
        
        print("\n180° rotation:")
        rotated180 = self._rotate_90(rotated90)
        self._print_board(rotated180)
        
        print("\n270° rotation:")
        rotated270 = self._rotate_90(rotated180)
        self._print_board(rotated270)
        
        print("\nHorizontal flip:")
        flipped_h = self._flip_horizontal(board)
        self._print_board(flipped_h)
        
        print("\nVertical flip:")
        flipped_v = self._flip_vertical(board)
        self._print_board(flipped_v)
        
        print("\nDiagonal flip:")
        flipped_d = self._flip_diagonal(board)
        self._print_board(flipped_d)
        
        print("\nAnti-diagonal flip:")
        flipped_ad = self._flip_antidiagonal(board)
        self._print_board(flipped_ad)

    def _print_board(self, board):
        """Helper to print a board in a readable format."""
        for row in board:
            line = ""
            for cell in row:
                if cell is None:
                    line += "_ "
                elif cell["type"] == "TIGER":
                    line += "T "
                else:
                    line += "G "
            print(line)
