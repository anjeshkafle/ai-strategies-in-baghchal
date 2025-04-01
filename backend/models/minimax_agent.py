from typing import List, Optional, Dict, Tuple
from models.game_state import GameState
from game_logic import get_all_possible_moves
import sys

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    """
    
    INF = 1000000
    
    def __init__(self, max_depth: int = 5, max_time: Optional[float] = None, randomize_equal_moves: bool = False,
                 max_table_size: int = 500000):
        self.max_depth = max_depth
        self.max_time = max_time  # Not used but kept for compatibility
        self.best_move = None
        self.best_score = None
        self.randomize_equal_moves = randomize_equal_moves  # Flag to control move randomization
        self.transposition_table = {}  # For storing evaluated positions
        self.inferred_moves = {}  # For tracking moves inferred from transposition table
        
        # Transposition table management parameters
        self.max_table_size = max_table_size  # Maximum number of entries
        self.tt_entries_by_depth = {}  # Track entries by depth for efficient pruning
        self.tt_size_check_frequency = 1000  # Check size every N insertions
        self.tt_insertion_count = 0  # Counter for insertions
        self.tt_replacement_percentage = 0.2  # Remove 20% of entries when table is full
        
        # Define all evaluation weights in one place for easy tuning
        # Mobility and space control
        self.mobility_weight_placement = 200     # Weight for movable tigers during placement
        self.mobility_weight_movement = 300      # Weight for movable tigers during movement
        self.closed_spaces_weight = 1000         # Weight for closed spaces (always 1000)
        
        # Capture-related weights
        self.base_capture_value = 3000           # Base value for each captured goat
        self.capture_speed_weight = 100000          # Weight for depth-sensitive capture bonus
        
        # Positioning weights
        self.dispersion_weight = 100             # Weight for tiger dispersion
        self.edge_weight = 300                   # Weight for goat edge preference
        
        # Debug mode flag
        self.debug_mode = True
    
    def evaluate(self, state: GameState, move_sequence=None) -> float:
        """
        Evaluates the current game state from Tiger's perspective using dynamic equilibrium points.
        Uses several core heuristics with weights and balance points that adapt to game progression:
        - Captures and threatened goats
        - Movable tigers and closed spaces
        - Tiger position, optimal spacing, and goat edge preference with dynamic equilibrium points
        """
        # Calculate traversed depth from move sequence
        traversed_depth = 0 if move_sequence is None else len(move_sequence)
            
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            final_score = MinimaxAgent.INF - traversed_depth  # Prefer faster wins
            return final_score
        elif winner == "GOAT":
            final_score = -MinimaxAgent.INF + traversed_depth  # Prefer slower losses from tiger's perspective
            return final_score
        
        # Compute the raw score based on board state and phase with dynamic equilibrium
        raw_score = self._compute_raw_score(state)
        
        # Adjust the score based on depth and captures
        final_score = self._adjust_score(raw_score, state, move_sequence)
        
        return final_score
    
    def evaluate_old(self, state: GameState, depth: int = 0) -> float:
        """
        [OLD VERSION - KEPT FOR REFERENCE]
        Evaluates the current game state from Tiger's perspective.
        Uses several core heuristics:
        - mobility_weight * movable_tigers (200 during placement, 300 during movement)
        - closed_spaces_weight * closed_spaces (1000)
        - base_capture_value * dead_goats + capture_speed_bonus
        - base_capture_value * threatened_goats (turn-dependent)
        - dispersion_weight * tiger_position_score (normalized 0-1 score)
        - optimal_spacing_weight * tiger_optimal_spacing (normalized 0-1 score)
        - -edge_weight * goat_edge_preference (normalized 0-1 score)
        """
        # Check for terminal states first
        winner = state.get_winner()
        if winner == "TIGER":
            final_score = MinimaxAgent.INF - depth  # Prefer faster wins
            return final_score
        elif winner == "GOAT":
            final_score = -MinimaxAgent.INF + depth  # Prefer slower losses from tiger's perspective
            return final_score
        
        # Compute the raw score based on board state and phase
        raw_score = self._compute_raw_score_old(state)
        
        # Adjust the score based on depth and captures
        final_score = self._adjust_score_old(raw_score, state, depth)
        
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
    
    def _compute_raw_score_old(self, state: GameState) -> float:
        """
        [OLD VERSION - KEPT FOR REFERENCE]
        Computes the raw evaluation score based solely on the board state and phase.
        This does not include depth penalty or capture bonus, which will be applied separately.
        """
        # Set mobility weight based on game phase
        mobility_weight = self.mobility_weight_placement if state.phase == "PLACEMENT" else self.mobility_weight_movement
        
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
        
        # Count closed spaces (positions where tigers are trapped)
        closed_regions = self._count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        closed_score = -self.closed_spaces_weight * total_closed_spaces
        score += closed_score
        
        # Calculate tiger positional score (normalized 0-1)
        position_score = self._calculate_tiger_positional_score(state)
        score += self.dispersion_weight * position_score
        
        # Calculate tiger optimal spacing score (normalized 0-1)
        # This heuristic is slightly more important than positional score
        optimal_spacing_score = self._calculate_tiger_optimal_spacing(state)
        optimal_spacing_weight = int(self.dispersion_weight * 1.5)  # 50% more weight than positional score
        score += optimal_spacing_weight * optimal_spacing_score
        
        # Calculate goat edge preference score (normalized 0-1)
        edge_score = self._calculate_goat_edge_preference(state)
        score -= self.edge_weight * edge_score  # Subtract from score (negative for tigers)
        
        return score
    
    def _adjust_score(self, raw_score: float, state: GameState, move_sequence=None) -> float:
        """
        Adjusts the raw evaluation score by applying depth penalty and dynamic capture bonuses.
        This version uses game-stage aware bonuses for captures.
        """
        # Start with the raw score
        adjusted_score = raw_score

        # Track capture depths
        capture_depths = []
        if move_sequence:
            for i, move in enumerate(move_sequence):
                if move.get("capture"):
                    capture_depths.append(i + 1)  # Depth starts from 1 (root is 0)

        # Calculate captures before search
        captures_before_search = state.goats_captured - len(capture_depths)

        # Calculate capture score with depth-proportional bonuses
        capture_score = captures_before_search * (self.base_capture_value + self.max_depth * self.capture_speed_weight)
        
        # Add depth-proportional bonus for each capture in the sequence
        for depth in capture_depths:
            capture_score += self.base_capture_value + (self.max_depth - depth) * self.capture_speed_weight

        # Debug log for visualization
        if self.debug_mode and move_sequence and capture_depths == [3,5]:
            print("\nMove sequence:", move_sequence)
            print("\nBoard state:")
            print("Captures during search:", len(capture_depths))
            print("Capture depths:", capture_depths)
            print("Captures before search:", captures_before_search)
            print("Capture score:", capture_score)
            for row in state.board:
                print("".join("_" if cell is None else cell["type"][0] for cell in row))

        # Depth traversed is the number of moves in the move sequence
        traversed_depth = len(move_sequence) if move_sequence else 0

        # Add capture score
        adjusted_score += capture_score
        # Apply depth penalty
        adjusted_score -= traversed_depth
        
        return adjusted_score
    
    def _adjust_score_old(self, raw_score: float, state: GameState, traversed_depth: int) -> float:
        """
        [OLD VERSION - KEPT FOR REFERENCE]
        Adjusts the raw evaluation score by applying depth penalty and capture bonuses.
        """
        # Start with the raw score
        adjusted_score = raw_score
        
        # Add capture score 
        capture_score = self.base_capture_value * state.goats_captured
        
        adjusted_score += capture_score
            
        # Apply depth penalty
        adjusted_score -= traversed_depth
        
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
        """Get the best move for the current state using minimax with alpha-beta pruning and transposition table."""
        # Reset transposition table and inferred moves for each new search
        self.transposition_table = {}
        self.inferred_moves = {}
        self.tt_entries_by_depth = {}
        self.tt_insertion_count = 0
        
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
            
            # Initialize move sequence with current move
            move_sequence = [move]
            
            next_is_max = next_state.turn == "TIGER"
            value = self.minimax(next_state, alpha, beta, next_is_max, move_sequence)
            
               
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
            print(f"Transposition table hits: {len(self.inferred_moves)}")
            print(f"Transposition table size: {len(self.transposition_table)}")
            
            # Print additional TT info
            tt_memory = sys.getsizeof(self.transposition_table)
            entries_by_depth = {d: len(entries) for d, entries in self.tt_entries_by_depth.items()}
            print(f"Transposition table memory usage: {tt_memory/1024:.2f} KB")
            print(f"Entries by depth: {entries_by_depth}")
            
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
            
            # Get a quick evaluation score
            score = self.evaluate(next_state)
            
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

    def minimax(self, state: GameState, alpha: float, beta: float, is_maximizing: bool, move_sequence=None):
        """Minimax algorithm with alpha-beta pruning and symmetry-aware transposition table."""
        # Initialize move_sequence if None
        if move_sequence is None:
            move_sequence = []
        
        # Calculate depths from move sequence
        traversed_depth = len(move_sequence)
        remaining_depth = self.max_depth - traversed_depth
            
        # Get canonical representation for transposition table lookup
        canonical_key, symmetry_type = self._get_canonical_state(state)
        tt_key = (canonical_key, remaining_depth, is_maximizing)
        
        # Check transposition table for exact match
        if tt_key in self.transposition_table:
            # For debugging, track when we use the transposition table at depth 1
            if remaining_depth == 1 and self.debug_mode:
                # Create a string representation of the state for debugging
                state_str = self._state_to_string(state)
                self.inferred_moves[state_str] = (self.transposition_table[tt_key], symmetry_type)
            
            return self.transposition_table[tt_key]
        
        # Base cases
        if remaining_depth == 0 or state.is_terminal():
            # Always evaluate from Tiger's perspective
            eval_score = self.evaluate(state, move_sequence)
            # Store exact evaluation in transposition table
            self._store_in_transposition_table(tt_key, eval_score, remaining_depth)
            return eval_score
        
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            eval_score = self.evaluate(state, move_sequence)
            self._store_in_transposition_table(tt_key, eval_score, remaining_depth)
            return eval_score
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(state, valid_moves)
        
        best_value = -MinimaxAgent.INF if is_maximizing else MinimaxAgent.INF
        
        for move in ordered_moves:
            new_state = state.clone()
            new_state.apply_move(move)
            
            # Add current move to sequence for child evaluation
            move_sequence.append(move)
            
            # Next turn alternates maximizing/minimizing
            next_is_max = new_state.turn == "TIGER"
            child_score = self.minimax(new_state, alpha, beta, next_is_max, move_sequence)
            
            # Remove the move after returning from recursive call
            move_sequence.pop()
            
            if is_maximizing:
                best_value = max(best_value, child_score)
                alpha = max(alpha, best_value)
            else:
                best_value = min(best_value, child_score)
                beta = min(beta, best_value)
                
            if beta <= alpha:
                break  # Alpha-beta pruning
        
        # Store the exact score (not bounds) in the transposition table
        self._store_in_transposition_table(tt_key, best_value, remaining_depth)
        return best_value
    
    def _store_in_transposition_table(self, tt_key, value, depth):
        """
        Store a value in the transposition table with size management.
        Prioritizes higher depth entries when the table needs pruning.
        """
        # Store the value
        self.transposition_table[tt_key] = value
        
        # Track the entry by depth for efficient pruning
        if depth not in self.tt_entries_by_depth:
            self.tt_entries_by_depth[depth] = set()
        self.tt_entries_by_depth[depth].add(tt_key)
        
        # Increment insertion counter
        self.tt_insertion_count += 1
        
        # Check if we need to manage table size
        if self.tt_insertion_count % self.tt_size_check_frequency == 0:
            self._manage_transposition_table_size()
    
    def _manage_transposition_table_size(self):
        """
        Manage the transposition table size by removing entries when it gets too large.
        Prioritizes keeping higher depth entries.
        """
        current_size = len(self.transposition_table)
        
        # If we're below the threshold, do nothing
        if current_size <= self.max_table_size:
            return
            
        # Calculate how many entries to remove (20% of max size)
        entries_to_remove = int(self.max_table_size * self.tt_replacement_percentage)
        
        # Get all depths, sorted from lowest to highest
        all_depths = sorted(self.tt_entries_by_depth.keys())
        
        # Remove entries starting from the lowest depths
        removed = 0
        for depth in all_depths:
            entries = self.tt_entries_by_depth[depth]
            entries_at_depth = len(entries)
            
            # If we can remove all entries at this depth
            if removed + entries_at_depth <= entries_to_remove:
                # Remove all entries at this depth
                for key in entries:
                    if key in self.transposition_table:
                        del self.transposition_table[key]
                # Clear the set at this depth
                self.tt_entries_by_depth[depth] = set()
                removed += entries_at_depth
            else:
                # Remove only as many as needed
                to_remove = entries_to_remove - removed
                keys_to_remove = list(entries)[:to_remove]
                
                for key in keys_to_remove:
                    if key in self.transposition_table:
                        del self.transposition_table[key]
                    entries.remove(key)
                
                removed += to_remove
                
            # If we've removed enough entries, stop
            if removed >= entries_to_remove:
                break
        
        if self.debug_mode:
            print(f"Pruned {removed} entries from transposition table. New size: {len(self.transposition_table)}")

    def _state_to_string(self, state: GameState) -> str:
        """Convert a game state to a string representation for debugging."""
        board_str = ""
        for row in state.board:
            for cell in row:
                if cell is None:
                    board_str += "_"
                elif cell["type"] == "TIGER":
                    board_str += "T"
                else:
                    board_str += "G"
        return f"{board_str}_{state.phase}_{state.turn}_{state.goats_placed}_{state.goats_captured}"
    
    def _board_to_string(self, board) -> str:
        """Convert just the board to a string representation."""
        board_str = ""
        for row in board:
            for cell in row:
                if cell is None:
                    board_str += "_"
                elif cell["type"] == "TIGER":
                    board_str += "T"
                else:
                    board_str += "G"
        return board_str
    
    def _get_canonical_state(self, state: GameState) -> Tuple[str, str]:
        """
        Get the canonical representation of the game state by trying all 8 symmetry transformations
        and returning the lexicographically smallest string representation.
        
        Returns:
            Tuple of (canonical_representation, symmetry_type)
        """
        board = state.board
        
        # Create a string representation of the original board
        original_str = self._board_to_string(board)
        
        # Initialize with original board
        canonical = original_str
        symmetry_type = "identity"
        
        # 90° rotation
        rotated90 = self._rotate_90(board)
        rotated90_str = self._board_to_string(rotated90)
        if rotated90_str < canonical:
            canonical = rotated90_str
            symmetry_type = "rotation_90"
        
        # 180° rotation
        rotated180 = self._rotate_90(rotated90)
        rotated180_str = self._board_to_string(rotated180)
        if rotated180_str < canonical:
            canonical = rotated180_str
            symmetry_type = "rotation_180"
        
        # 270° rotation
        rotated270 = self._rotate_90(rotated180)
        rotated270_str = self._board_to_string(rotated270)
        if rotated270_str < canonical:
            canonical = rotated270_str
            symmetry_type = "rotation_270"
        
        # Horizontal flip
        flipped_h = self._flip_horizontal(board)
        flipped_h_str = self._board_to_string(flipped_h)
        if flipped_h_str < canonical:
            canonical = flipped_h_str
            symmetry_type = "flip_horizontal"
        
        # Vertical flip
        flipped_v = self._flip_vertical(board)
        flipped_v_str = self._board_to_string(flipped_v)
        if flipped_v_str < canonical:
            canonical = flipped_v_str
            symmetry_type = "flip_vertical"
        
        # Diagonal flip (top-left to bottom-right)
        flipped_d = self._flip_diagonal(board)
        flipped_d_str = self._board_to_string(flipped_d)
        if flipped_d_str < canonical:
            canonical = flipped_d_str
            symmetry_type = "flip_diagonal"
        
        # Anti-diagonal flip (top-right to bottom-left)
        flipped_ad = self._flip_antidiagonal(board)
        flipped_ad_str = self._board_to_string(flipped_ad)
        if flipped_ad_str < canonical:
            canonical = flipped_ad_str
            symmetry_type = "flip_antidiagonal"
        
        # Add game state info to the canonical representation
        phase_info = f"_{state.phase}_{state.goats_placed}_{state.goats_captured}_{state.turn}"
        return canonical + phase_info, symmetry_type
    
    def _rotate_90(self, board):
        """Rotate the board 90 degrees clockwise."""
        n = len(board)
        result = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[j][n-1-i] = board[i][j]
        return result
    
    def _flip_horizontal(self, board):
        """Flip the board horizontally (around the x-axis)."""
        n = len(board)
        result = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[n-1-i][j] = board[i][j]
        return result
    
    def _flip_vertical(self, board):
        """Flip the board vertically (around the y-axis)."""
        n = len(board)
        result = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[i][n-1-j] = board[i][j]
        return result
    
    def _flip_diagonal(self, board):
        """Flip the board across the main diagonal (top-left to bottom-right)."""
        n = len(board)
        result = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[j][i] = board[i][j]
        return result
    
    def _flip_antidiagonal(self, board):
        """Flip the board across the anti-diagonal (top-right to bottom-left)."""
        n = len(board)
        result = [[None for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                result[n-1-j][n-1-i] = board[i][j]
        return result
    
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