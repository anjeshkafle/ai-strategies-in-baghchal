from typing import List, Optional, Dict, Tuple
import random
import math
import json
import os
import time
from copy import deepcopy
from models.game_state import GameState
from game_logic import get_all_possible_moves, get_threatened_nodes

# Shared utility functions for feature extraction
def _count_movable_tigers(all_tiger_moves) -> int:
    """
    Counts the number of tigers that have at least one valid move.
    """
    # Count tigers with at least one move
    movable_tigers = set()
    for move in all_tiger_moves:
        from_pos = (move["from"]["x"], move["from"]["y"])
        movable_tigers.add(from_pos)
    
    return len(movable_tigers)

def _count_closed_spaces(state: GameState, all_tiger_moves) -> List[List[tuple[int, int]]]:
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
    
    # Directly iterate through the board
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
                        _is_valid_connection(curr_x, curr_y, new_x, new_y)):
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

def _count_threatened_goats(threatened_data, board) -> int:
    """
    Count the number of goats that are actually threatened (can be captured).
    Uses the threatened_data from state.get_threatened_nodes().
    
    Returns:
        The count of threatened goats.
    """
    threatened_count = 0
    
    # Track which goat positions have been counted
    counted_positions = set()
    
    for goat_x, goat_y, landing_x, landing_y in threatened_data:
        # Check if there's a goat at the threatened position
        if (board[goat_y][goat_x] is not None and 
            board[goat_y][goat_x]["type"] == "GOAT" and
            # Check if the landing position is empty
            board[landing_y][landing_x] is None and
            # Make sure we haven't already counted this goat
            (goat_x, goat_y) not in counted_positions):
            
            threatened_count += 1
            counted_positions.add((goat_x, goat_y))
    
    return threatened_count

def _is_valid_connection(from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
    """Helper method to check if two positions are validly connected on the board."""
    # Orthogonal moves are always valid if adjacent
    if abs(from_x - to_x) + abs(from_y - to_y) == 1:
        return True

    # Diagonal moves need special handling
    if abs(from_x - to_x) == 1 and abs(from_y - to_y) == 1:
        # No diagonal moves for second and fourth nodes on outer edges
        if _is_outer_layer(from_x, from_y):
            is_second_or_fourth_node = (
                ((from_x == 0 or from_x == 4) and (from_y == 1 or from_y == 3)) or
                ((from_y == 0 or from_y == 4) and (from_x == 1 or from_x == 3))
            )
            if is_second_or_fourth_node:
                return False

        # No diagonal moves for middle nodes in second layer
        if _is_second_layer(from_x, from_y):
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

def _is_outer_layer(x: int, y: int) -> bool:
    """Check if a position is on the outer layer of the board."""
    return x == 0 or y == 0 or x == 4 or y == 4

def _is_second_layer(x: int, y: int) -> bool:
    """Check if a position is on the second layer of the board."""
    return x == 1 or y == 1 or x == 3 or y == 3

def _calculate_tiger_positional_score(state: GameState) -> float:
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

def _calculate_tiger_optimal_spacing(state: GameState) -> float:
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
                    _is_valid_connection(x1, y1, middle_x1, y1)):
                    optimal_pairs += 1
                    continue
            
            # Same column
            if x1 == x2 and abs(y1 - y2) == 3:
                # Check if both intermediary nodes are empty
                middle_y1 = min(y1, y2) + 1
                middle_y2 = min(y1, y2) + 2
                
                if (state.board[middle_y1][x1] is None and 
                    state.board[middle_y2][x1] is None and
                    _is_valid_connection(x1, y1, x1, middle_y1)):
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
                    if _is_valid_connection(x1, y1, middle_x1, middle_y1):
                        optimal_pairs += 1
                        continue
    
    # Normalize based on total possible pairs
    if total_pairs > 0:
        normalized_score = optimal_pairs / total_pairs
    else:
        normalized_score = 0
        
    return normalized_score
    
def _calculate_goat_edge_preference(state: GameState) -> float:
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
                if _is_outer_layer(x, y):
                    outer_layer_goats += 1
                elif _is_second_layer(x, y):
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
    
    # Apply a severe penalty for center goats in early game (first 5 goats)
    if total_goats <= 5 and center_goats > 0:
        # Penalty proportional to center goats and inversely proportional to total goats
        center_penalty = min(0.5, (center_goats / total_goats) * 0.6)
        placement_quality -= center_penalty
        
        # Ensure we don't go below 0
        placement_quality = max(0.0, placement_quality)
    
    return placement_quality

class TigerAgent:
    """
    Q-learning agent for the Tiger player in Bagh Chal.
    Maintains its own Q-table and visit counts.
    """
    
    def __init__(self, discount_factor=0.95):
        # Q-table mapping state features to action values
        self.q_table = {}  # Dict[state_tuple, Dict[action_name, value]]
        
        # Visit counts for adaptive learning rates
        self.visit_counts = {}  # Dict[state_tuple, Dict[action_name, count]]
        
        # Configuration
        self.discount_factor = discount_factor
        
        # For action selection
        self._chosen_concrete_move = None
    
    def get_move(self, state: GameState, epsilon=0.0) -> Dict:
        """
        Choose a move based on the current state.
        
        Args:
            state: Current game state
            epsilon: Exploration rate (0.0 for inference)
            
        Returns:
            A concrete move to apply
        """
        # Choose abstract action
        abstract_action = self._choose_action(state, epsilon)
        if abstract_action is None:
            return None
            
        # Get concrete move for the chosen action
        return self._get_concrete_move(state, abstract_action)
    
    def _choose_action(self, state: GameState, epsilon: float) -> str:
        """Choose an action using epsilon-greedy strategy with tie-breaking"""
        # Get state features
        s = self._get_state_features(state)
        
        # Get available abstract actions
        abstract_actions = self._get_abstract_actions(state)
        
        # If no valid actions, return None
        if not abstract_actions:
            return None
        
        # Exploration: random action with probability epsilon
        if random.random() < epsilon:
            return random.choice(abstract_actions)
        
        # Exploitation: choose best action based on Q-values
        q_values = self.q_table.get(s, {})
        
        # Rank actions by Q-value, with default of 0.0 for unseen actions
        ranked_actions = sorted(
            abstract_actions,
            key=lambda a: q_values.get(a, 0.0),
            reverse=True
        )
        
        # Try each action in order of Q-value
        for current_action in ranked_actions:
            candidate_moves = self._get_candidate_moves(current_action, state)
            if candidate_moves:
                if len(candidate_moves) == 1:
                    # Only one move for this action
                    self._chosen_concrete_move = candidate_moves[0]
                    return current_action
                else:
                    # Multiple candidates, try to refine using next action in ranking
                    next_action_index = ranked_actions.index(current_action) + 1
                    
                    if next_action_index < len(ranked_actions):
                        next_action = ranked_actions[next_action_index]
                        next_candidates = self._get_candidate_moves(next_action, state)
                        
                        # Find common moves between current and next action
                        common_moves = [m for m in candidate_moves if m in next_candidates]
                        
                        if common_moves:
                            # Choose randomly from refined candidates
                            self._chosen_concrete_move = random.choice(common_moves)
                            return current_action
                    
                    # No refinement possible, choose randomly from current candidates
                    self._chosen_concrete_move = random.choice(candidate_moves)
                    return current_action
        
        # No valid action found
        return None
    
    def _get_abstract_actions(self, state: GameState) -> list:
        """Get available abstract actions for the tiger player"""
        available_actions = []
        
        # Only consider tiger actions if it's tiger's turn
        if state.turn != "TIGER":
            return []
        
        # Always consider Capture_Goat if any captures are available
        all_moves = state.get_valid_moves()
        capture_moves = [move for move in all_moves if move.get("capture")]
        
        if capture_moves:
            available_actions.append("Capture_Goat")
        
        # Always consider Improve_Position
        available_actions.append("Improve_Position")
        
        # Consider Improve_Spacing
        available_actions.append("Improve_Spacing")
        
        # Consider Increase_Mobility
        available_actions.append("Increase_Mobility")
        
        # Consider Setup_Threat
        available_actions.append("Setup_Threat")
        
        # Always consider Safe_Move as fallback
        available_actions.append("Safe_Move")
        
        return available_actions
    
    def _get_concrete_move(self, state: GameState, action: str) -> Dict:
        """Return the concrete move for the chosen abstract action"""
        # If we've already chosen a concrete move during action selection, return it
        if self._chosen_concrete_move is not None:
            move = self._chosen_concrete_move
            self._chosen_concrete_move = None  # Reset for next time
            return move
        
        # Otherwise, choose a move based on the abstract action
        candidates = self._get_candidate_moves(action, state)
        if candidates:
            return random.choice(candidates)
        
        # Last resort: return any valid move
        valid_moves = state.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None
    
    def _get_candidate_moves(self, abstract_action: str, state: GameState) -> list:
        """Get concrete moves corresponding to the given abstract action"""
        all_moves = state.get_valid_moves()
        
        if not all_moves:
            return []
        
        # Handle specific abstract actions
        if abstract_action == "Capture_Goat":
            return [move for move in all_moves if move.get("capture")]
        
        if abstract_action == "Improve_Position":
            # Rank moves by position improvement
            return self._rank_moves_by_improvement(state, all_moves, self._evaluate_position_improvement)
        
        if abstract_action == "Improve_Spacing":
            # Rank moves by spacing improvement
            return self._rank_moves_by_improvement(state, all_moves, self._evaluate_spacing_improvement)
        
        if abstract_action == "Increase_Mobility":
            # Rank moves by mobility improvement
            return self._rank_moves_by_improvement(state, all_moves, self._evaluate_mobility_improvement)
        
        if abstract_action == "Setup_Threat":
            # Find moves that create future capture opportunities
            return self._find_threat_setup_moves(state, all_moves)
        
        # Safe_Move: any move that doesn't worsen the position significantly
        return all_moves
    
    def _rank_moves_by_improvement(self, state: GameState, moves: list, evaluation_func) -> list:
        """
        Rank moves by how much they improve the position according to the given evaluation function.
        Returns all moves that don't worsen the position.
        """
        move_scores = []
        
        for move in moves:
            # Skip moves that we already know are captures
            if move.get("capture"):
                continue
                
            # Apply the move to a cloned state
            next_state = state.clone()
            next_state.apply_move(move)
            
            # Evaluate the improvement
            improvement = evaluation_func(state, next_state)
            
            # Add to our list of scored moves
            move_scores.append((move, improvement))
        
        # Filter to moves that don't worsen the position
        non_worsening_moves = [move for move, score in move_scores if score >= 0]
        
        # If we have non-worsening moves, return them
        if non_worsening_moves:
            return non_worsening_moves
        
        # Otherwise, return all moves as a fallback
        return [move for move, _ in move_scores]
    
    def _evaluate_position_improvement(self, old_state: GameState, new_state: GameState) -> float:
        """Evaluate how much a move improves tiger positioning"""
        old_score = _calculate_tiger_positional_score(old_state)
        new_score = _calculate_tiger_positional_score(new_state)
        return new_score - old_score
    
    def _evaluate_spacing_improvement(self, old_state: GameState, new_state: GameState) -> float:
        """Evaluate how much a move improves tiger spacing"""
        old_score = _calculate_tiger_optimal_spacing(old_state)
        new_score = _calculate_tiger_optimal_spacing(new_state)
        return new_score - old_score
    
    def _evaluate_mobility_improvement(self, old_state: GameState, new_state: GameState) -> float:
        """Evaluate how much a move improves tiger mobility"""
        old_moves = get_all_possible_moves(old_state.board, "MOVEMENT", "TIGER")
        new_moves = get_all_possible_moves(new_state.board, "MOVEMENT", "TIGER")
        
        # Calculate the change in total number of moves
        old_mobility = len(old_moves)
        new_mobility = len(new_moves)
        
        # Normalize the change to a range of -1 to 1
        max_possible_improvement = 10  # Assuming maximum improvement is around 10 moves
        normalized_change = (new_mobility - old_mobility) / max_possible_improvement
        
        return normalized_change
    
    def _find_threat_setup_moves(self, state: GameState, moves: list) -> list:
        """Find moves that create new capture opportunities"""
        setup_moves = []
        
        # Get current capture opportunities
        current_capture_moves = [m for m in get_all_possible_moves(state.board, "MOVEMENT", "TIGER") if m.get("capture")]
        current_capturable_goats = {(m["capture"]["x"], m["capture"]["y"]) for m in current_capture_moves}
        
        for move in moves:
            # Skip moves that are already captures
            if move.get("capture"):
                continue
                
            # Apply the move to a cloned state
            next_state = state.clone()
            next_state.apply_move(move)
            
            # Get capture opportunities after the move
            next_capture_moves = [m for m in get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER") if m.get("capture")]
            next_capturable_goats = {(m["capture"]["x"], m["capture"]["y"]) for m in next_capture_moves}
            
            # If there are new capture opportunities, add this move to setup_moves
            if len(next_capturable_goats) > len(current_capturable_goats):
                setup_moves.append(move)
        
        return setup_moves
    
    def _get_state_features(self, state: GameState) -> tuple:
        """Extract features from game state and bucketize them"""
        # Get the raw metrics from the state
        goats_captured = state.goats_captured
        goats_placed = state.goats_placed
        goats_on_board = goats_placed - goats_captured
        
        # Get all tiger moves for various metrics
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Calculate movable tigers
        movable_tigers = _count_movable_tigers(all_tiger_moves)
        
        # Calculate tiger mobility
        tiger_mobility = len(all_tiger_moves)
        
        # Calculate goat mobility
        all_goat_moves = get_all_possible_moves(state.board, state.phase, "GOAT")
        goat_mobility = len(all_goat_moves)
        
        # Calculate immediate captures available
        immediate_captures = sum(1 for move in all_tiger_moves if move.get("capture"))
        
        # Get threatened nodes data
        threatened_data = state.get_threatened_nodes()
        
        # Count threatened goats
        threatened_goats_count = _count_threatened_goats(threatened_data, state.board)
        
        # Calculate closed spaces
        closed_regions = _count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Calculate tiger positional score
        tiger_position_score = _calculate_tiger_positional_score(state)
        
        # Calculate tiger spacing score
        tiger_spacing_score = _calculate_tiger_optimal_spacing(state)
        
        # Calculate goat edge preference
        goat_edge_score = _calculate_goat_edge_preference(state)
        
        # Count trapped tigers (tigers with no moves)
        total_tigers = 4
        tigers_trapped = total_tigers - movable_tigers
        
        # Bucketize the features
        goats_captured_bucket = min(5, goats_captured)
        tigers_trapped_bucket = min(4, tigers_trapped)
        
        # Tiger mobility buckets
        if tiger_mobility <= 4:
            tiger_mobility_bucket = "Low:0-4"
        elif tiger_mobility <= 9:
            tiger_mobility_bucket = "Med:5-9"
        else:
            tiger_mobility_bucket = "High:10+"
        
        # Goat mobility buckets
        if goat_mobility <= 10:
            goat_mobility_bucket = "Low:0-10"
        elif goat_mobility <= 20:
            goat_mobility_bucket = "Med:11-20"
        else:
            goat_mobility_bucket = "High:21+"
        
        # Immediate captures buckets
        if immediate_captures == 0:
            captures_bucket = 0
        elif immediate_captures == 1:
            captures_bucket = 1
        else:
            captures_bucket = "2+"
        
        # Setup threats (threatened goats minus immediate captures)
        setup_threats = threatened_goats_count - immediate_captures
        if setup_threats == 0:
            threats_bucket = "None:0"
        elif setup_threats <= 2:
            threats_bucket = "Low:1-2"
        else:
            threats_bucket = "High:3+"
        
        # Tiger position score buckets
        if tiger_position_score < 0.3:
            position_bucket = "Low:0-0.3"
        elif tiger_position_score < 0.7:
            position_bucket = "Med:0.3-0.7"
        else:
            position_bucket = "High:0.7-1.0"
        
        # Tiger spacing score buckets
        if tiger_spacing_score < 0.3:
            spacing_bucket = "Low:0-0.3"
        elif tiger_spacing_score < 0.7:
            spacing_bucket = "Med:0.3-0.7"
        else:
            spacing_bucket = "High:0.7-1.0"
        
        # Goat edge score buckets
        if goat_edge_score < 0.4:
            edge_bucket = "Low:0-0.4"
        elif goat_edge_score < 0.7:
            edge_bucket = "Med:0.4-0.7"
        else:
            edge_bucket = "High:0.7-1.0"
        
        # Closed spaces buckets
        if total_closed_spaces == 0:
            closed_bucket = "None:0"
        elif total_closed_spaces <= 5:
            closed_bucket = "Low:1-5"
        else:
            closed_bucket = "High:6+"
        
        # Goats on board buckets
        if goats_on_board <= 8:
            goats_bucket = "Few:0-8"
        elif goats_on_board <= 15:
            goats_bucket = "Mid:9-15"
        else:
            goats_bucket = "Many:16-20"
        
        # Game phase
        phase = state.phase
        
        # Return the state features as a tuple for hash-ability
        return (
            goats_captured_bucket,
            tigers_trapped_bucket,
            tiger_mobility_bucket,
            goat_mobility_bucket,
            captures_bucket,
            threats_bucket,
            position_bucket,
            spacing_bucket,
            edge_bucket,
            closed_bucket,
            goats_bucket,
            phase
        )
    
    def _get_reward(self, prev_state: GameState, action: str, next_state: GameState) -> float:
        """Calculate rewards for tiger actions"""
        # Check for terminal states first
        winner = next_state.get_winner()
        if winner == "TIGER":
            return 100.0
        elif winner == "GOAT":
            return -100.0
        
        # Base reward (small penalty per move to encourage shorter solutions)
        reward = -0.1
        
        # Reward for captures
        if next_state.goats_captured > prev_state.goats_captured:
            reward += 25.0
        
        # Reward for position improvement
        old_position_score = _calculate_tiger_positional_score(prev_state)
        new_position_score = _calculate_tiger_positional_score(next_state)
        position_improvement = new_position_score - old_position_score
        if position_improvement > 0:
            reward += 5.0 * position_improvement
        
        # Reward for spacing improvement
        old_spacing_score = _calculate_tiger_optimal_spacing(prev_state)
        new_spacing_score = _calculate_tiger_optimal_spacing(next_state)
        spacing_improvement = new_spacing_score - old_spacing_score
        if spacing_improvement > 0:
            reward += 5.0 * spacing_improvement
        
        # Reward for creating threats
        prev_threats = len([m for m in get_all_possible_moves(prev_state.board, "MOVEMENT", "TIGER") if m.get("capture")])
        next_threats = len([m for m in get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER") if m.get("capture")])
        if next_threats > prev_threats:
            reward += 5.0
        
        return reward
    
    def update_q_table(self, state: GameState, action: str, reward: float, next_state: GameState):
        """Update Q-table and visit counts using adaptive learning rate"""
        # Get state features
        s = self._get_state_features(state)
        s_next = self._get_state_features(next_state)
        
        # Initialize Q-values if needed
        if s not in self.q_table:
            self.q_table[s] = {}
        if s not in self.visit_counts:
            self.visit_counts[s] = {}
        
        # Initialize action if needed
        if action not in self.q_table[s]:
            self.q_table[s][action] = 0.0
        if action not in self.visit_counts[s]:
            self.visit_counts[s][action] = 0
        
        # Increment visit count
        self.visit_counts[s][action] += 1
        
        # Calculate adaptive learning rate
        visit_count = self.visit_counts[s][action]
        alpha = 1.0 / (1.0 + visit_count)  # More stable adaptive learning rate
        
        # Get current Q-value
        current_q = self.q_table[s][action]
        
        # Get maximum Q-value for next state
        next_q_values = self.q_table.get(s_next, {})
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Update Q-value using Bellman equation with adaptive learning rate
        self.q_table[s][action] = current_q + alpha * (reward + self.discount_factor * max_next_q - current_q)


class GoatAgent:
    """
    Q-learning agent for the Goat player in Bagh Chal.
    Maintains its own Q-table and visit counts.
    """
    
    def __init__(self, discount_factor=0.95):
        # Q-table mapping state features to action values
        self.q_table = {}  # Dict[state_tuple, Dict[action_name, value]]
        
        # Visit counts for adaptive learning rates
        self.visit_counts = {}  # Dict[state_tuple, Dict[action_name, count]]
        
        # Configuration
        self.discount_factor = discount_factor
        
        # For action selection
        self._chosen_concrete_move = None
    
    def get_move(self, state: GameState, epsilon=0.0) -> Dict:
        """
        Choose a move based on the current state.
        
        Args:
            state: Current game state
            epsilon: Exploration rate (0.0 for inference)
            
        Returns:
            A concrete move to apply
        """
        # Choose abstract action
        abstract_action = self._choose_action(state, epsilon)
        if abstract_action is None:
            return None
            
        # Get concrete move for the chosen action
        return self._get_concrete_move(state, abstract_action)
    
    def _choose_action(self, state: GameState, epsilon: float) -> str:
        """Choose an action using epsilon-greedy strategy with tie-breaking"""
        # Get state features
        s = self._get_state_features(state)
        
        # Get available abstract actions
        abstract_actions = self._get_abstract_actions(state)
        
        # If no valid actions, return None
        if not abstract_actions:
            return None
        
        # Exploration: random action with probability epsilon
        if random.random() < epsilon:
            return random.choice(abstract_actions)
        
        # Exploitation: choose best action based on Q-values
        q_values = self.q_table.get(s, {})
        
        # Rank actions by Q-value, with default of 0.0 for unseen actions
        ranked_actions = sorted(
            abstract_actions,
            key=lambda a: q_values.get(a, 0.0),
            reverse=True
        )
        
        # Try each action in order of Q-value
        for current_action in ranked_actions:
            candidate_moves = self._get_candidate_moves(current_action, state)
            if candidate_moves:
                if len(candidate_moves) == 1:
                    # Only one move for this action
                    self._chosen_concrete_move = candidate_moves[0]
                    return current_action
                else:
                    # Multiple candidates, try to refine using next action in ranking
                    next_action_index = ranked_actions.index(current_action) + 1
                    
                    if next_action_index < len(ranked_actions):
                        next_action = ranked_actions[next_action_index]
                        next_candidates = self._get_candidate_moves(next_action, state)
                        
                        # Find common moves between current and next action
                        common_moves = [m for m in candidate_moves if m in next_candidates]
                        
                        if common_moves:
                            # Choose randomly from refined candidates
                            self._chosen_concrete_move = random.choice(common_moves)
                            return current_action
                    
                    # No refinement possible, choose randomly from current candidates
                    self._chosen_concrete_move = random.choice(candidate_moves)
                    return current_action
        
        # No valid action found
        return None
    
    def _get_abstract_actions(self, state: GameState) -> list:
        """Get available abstract actions for the goat player"""
        available_actions = []
        
        # Only consider goat actions if it's goat's turn
        if state.turn != "GOAT":
            return []
        
        # Get all threatened nodes data
        threatened_data = state.get_threatened_nodes()
        
        # Actions common to both phases
        if threatened_data:
            if state.phase == "PLACEMENT":
                available_actions.append("Block_Immediate_Capture")
                available_actions.append("Safe_Placement")
            else:  # MOVEMENT
                available_actions.append("Block_Immediate_Capture")
                available_actions.append("Escape_Threat")
                available_actions.append("Safe_Move")
        
        # Phase-specific actions
        if state.phase == "PLACEMENT":
            # For placement, focus on edge positions and reducing tiger mobility
            available_actions.append("Improve_Edge_Position")
            available_actions.append("Reduce_Tiger_Mobility")
            available_actions.append("Safe_Placement")
        else:  # MOVEMENT
            # For movement, focus on edge positions, reducing tiger mobility, 
            # and contributing to closed spaces
            available_actions.append("Improve_Edge_Position")
            available_actions.append("Reduce_Tiger_Mobility")
            available_actions.append("Contribute_To_Trap")
            available_actions.append("Safe_Move")
        
        return available_actions
    
    def _get_concrete_move(self, state: GameState, action: str) -> Dict:
        """Return the concrete move for the chosen abstract action"""
        # If we've already chosen a concrete move during action selection, return it
        if self._chosen_concrete_move is not None:
            move = self._chosen_concrete_move
            self._chosen_concrete_move = None  # Reset for next time
            return move
        
        # Otherwise, choose a move based on the abstract action
        candidates = self._get_candidate_moves(action, state)
        if candidates:
            return random.choice(candidates)
        
        # Last resort: return any valid move
        valid_moves = state.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None
    
    def _get_candidate_moves(self, abstract_action: str, state: GameState) -> list:
        """Get concrete moves corresponding to the given abstract action"""
        all_moves = state.get_valid_moves()
        
        if not all_moves:
            return []
        
        # Get threatened data
        threatened_data = state.get_threatened_nodes()
        
        # Create lookup dictionaries for faster checking
        hot_squares = {}  # key: (x, y), value: list of (landing_x, landing_y)
        landing_squares = {}  # key: (landing_x, landing_y), value: list of (x, y)
        
        for goat_x, goat_y, landing_x, landing_y in threatened_data:
            # Add to hot_squares dictionary
            if (goat_x, goat_y) not in hot_squares:
                hot_squares[(goat_x, goat_y)] = []
            hot_squares[(goat_x, goat_y)].append((landing_x, landing_y))
            
            # Add to landing_squares dictionary
            if (landing_x, landing_y) not in landing_squares:
                landing_squares[(landing_x, landing_y)] = []
            landing_squares[(landing_x, landing_y)].append((goat_x, goat_y))
        
        # Handle specific abstract actions
        if abstract_action == "Block_Immediate_Capture":
            # Find moves that block a tiger's landing position
            return self._find_blocking_moves(state, all_moves, landing_squares, hot_squares)
        
        if abstract_action == "Escape_Threat":
            # Find moves that escape from a threatened position
            return self._find_escape_moves(state, all_moves, hot_squares)
        
        if abstract_action == "Improve_Edge_Position":
            # Moves that improve edge positioning
            return self._rank_moves_by_improvement(state, all_moves, self._evaluate_edge_improvement)
        
        if abstract_action == "Reduce_Tiger_Mobility":
            # Moves that reduce tiger mobility
            return self._rank_moves_by_improvement(state, all_moves, self._evaluate_mobility_reduction)
        
        if abstract_action == "Contribute_To_Trap":
            # Moves that help create or maintain tiger traps
            return self._find_trap_contribution_moves(state, all_moves)
        
        # Safe_Placement/Safe_Move: any move that doesn't place/move to a threatened position
        if "Safe" in abstract_action:
            return self._find_safe_moves(state, all_moves, hot_squares)
        
        # Fallback - return all moves
        return all_moves
    
    def _find_blocking_moves(self, state: GameState, moves: list, landing_squares: dict, hot_squares: dict) -> list:
        """Find moves that block an immediate capture"""
        blocking_moves = []
        
        for move in moves:
            # Determine target position based on move type
            if state.phase == "PLACEMENT":
                target_x, target_y = move["x"], move["y"]
            else:  # MOVEMENT
                target_x, target_y = move["to"]["x"], move["to"]["y"]
            
            # Check if this position is in the landing_squares dictionary
            if (target_x, target_y) in landing_squares:
                # For each threatened goat that this move could protect
                for goat_x, goat_y in landing_squares[(target_x, target_y)]:
                    # Check if there's actually a goat at that position
                    if (state.board[goat_y][goat_x] is not None and 
                        state.board[goat_y][goat_x]["type"] == "GOAT"):
                        
                        # In movement phase, make sure we're not moving the threatened goat itself
                        if state.phase == "MOVEMENT" and move["from"]["x"] == goat_x and move["from"]["y"] == goat_y:
                            continue
                        
                        # This move blocks a capture
                        blocking_moves.append(move)
                        break  # Once we know it's a blocking move, no need to check other goats
        
        return blocking_moves
    
    def _find_escape_moves(self, state: GameState, moves: list, hot_squares: dict) -> list:
        """Find moves that escape from threatened positions"""
        escape_moves = []
        
        # Only applicable in movement phase
        if state.phase != "MOVEMENT":
            return []
        
        for move in moves:
            from_x, from_y = move["from"]["x"], move["from"]["y"]
            to_x, to_y = move["to"]["x"], move["to"]["y"]
            
            # Check if the from position is threatened
            if (from_x, from_y) in hot_squares:
                # Check if the to position is safe
                is_to_safe = (to_x, to_y) not in hot_squares
                
                if is_to_safe:
                    escape_moves.append(move)
        
        return escape_moves
    
    def _rank_moves_by_improvement(self, state: GameState, moves: list, evaluation_func) -> list:
        """
        Rank moves by how much they improve the position according to the given evaluation function.
        Returns all moves that don't worsen the position.
        """
        move_scores = []
        
        for move in moves:
            # Apply the move to a cloned state
            next_state = state.clone()
            next_state.apply_move(move)
            
            # Evaluate the improvement
            improvement = evaluation_func(state, next_state)
            
            # Add to our list of scored moves
            move_scores.append((move, improvement))
        
        # Filter to moves that don't worsen the position
        non_worsening_moves = [move for move, score in move_scores if score >= 0]
        
        # If we have non-worsening moves, return them
        if non_worsening_moves:
            return non_worsening_moves
        
        # Otherwise, return all moves as a fallback
        return [move for move, _ in move_scores]
    
    def _evaluate_edge_improvement(self, old_state: GameState, new_state: GameState) -> float:
        """Evaluate how much a move improves goat edge positioning"""
        old_score = _calculate_goat_edge_preference(old_state)
        new_score = _calculate_goat_edge_preference(new_state)
        return new_score - old_score
    
    def _evaluate_mobility_reduction(self, old_state: GameState, new_state: GameState) -> float:
        """Evaluate how much a move reduces tiger mobility"""
        old_moves = get_all_possible_moves(old_state.board, "MOVEMENT", "TIGER")
        new_moves = get_all_possible_moves(new_state.board, "MOVEMENT", "TIGER")
        
        # Calculate the change in total number of moves
        old_mobility = len(old_moves)
        new_mobility = len(new_moves)
        
        # We want to reduce tiger mobility, so a negative change is good
        # Normalize the change to a range of -1 to 1
        max_possible_reduction = 10  # Assuming maximum reduction is around 10 moves
        normalized_change = (old_mobility - new_mobility) / max_possible_reduction
        
        return normalized_change
    
    def _find_trap_contribution_moves(self, state: GameState, moves: list) -> list:
        """Find moves that help create or maintain tiger traps"""
        trap_moves = []
        
        # Only relevant in movement phase
        if state.phase != "MOVEMENT":
            return []
        
        # Get all tiger moves
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Get current closed spaces
        current_closed_regions = _count_closed_spaces(state, all_tiger_moves)
        current_total_closed = sum(len(region) for region in current_closed_regions)
        
        for move in moves:
            # Apply the move to a cloned state
            next_state = state.clone()
            next_state.apply_move(move)
            
            # Get all tiger moves in the new state
            next_tiger_moves = get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER")
            
            # Get closed spaces after the move
            next_closed_regions = _count_closed_spaces(next_state, next_tiger_moves)
            next_total_closed = sum(len(region) for region in next_closed_regions)
            
            # If there are more closed spaces or trapped tigers, this move contributes to trapping
            if next_total_closed > current_total_closed:
                trap_moves.append(move)
        
        return trap_moves
    
    def _find_safe_moves(self, state: GameState, moves: list, hot_squares: dict) -> list:
        """Find moves that don't place/move to a threatened position"""
        safe_moves = []
        
        for move in moves:
            # Determine target position based on move type
            if state.phase == "PLACEMENT":
                target_x, target_y = move["x"], move["y"]
            else:  # MOVEMENT
                target_x, target_y = move["to"]["x"], move["to"]["y"]
            
            # Check if the target position is safe
            is_target_safe = (target_x, target_y) not in hot_squares
            
            if is_target_safe:
                safe_moves.append(move)
        
        return safe_moves
    
    def _get_state_features(self, state: GameState) -> tuple:
        """Extract features from game state and bucketize them"""
        # This is the same as TigerAgent._get_state_features
        # Get the raw metrics from the state
        goats_captured = state.goats_captured
        goats_placed = state.goats_placed
        goats_on_board = goats_placed - goats_captured
        
        # Get all tiger moves for various metrics
        all_tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
        
        # Calculate movable tigers
        movable_tigers = _count_movable_tigers(all_tiger_moves)
        
        # Calculate tiger mobility
        tiger_mobility = len(all_tiger_moves)
        
        # Calculate goat mobility
        all_goat_moves = get_all_possible_moves(state.board, state.phase, "GOAT")
        goat_mobility = len(all_goat_moves)
        
        # Calculate immediate captures available
        immediate_captures = sum(1 for move in all_tiger_moves if move.get("capture"))
        
        # Get threatened nodes data
        threatened_data = state.get_threatened_nodes()
        
        # Count threatened goats
        threatened_goats_count = _count_threatened_goats(threatened_data, state.board)
        
        # Calculate closed spaces
        closed_regions = _count_closed_spaces(state, all_tiger_moves)
        total_closed_spaces = sum(len(region) for region in closed_regions)
        
        # Calculate tiger positional score
        tiger_position_score = _calculate_tiger_positional_score(state)
        
        # Calculate tiger spacing score
        tiger_spacing_score = _calculate_tiger_optimal_spacing(state)
        
        # Calculate goat edge preference
        goat_edge_score = _calculate_goat_edge_preference(state)
        
        # Count trapped tigers (tigers with no moves)
        total_tigers = 4
        tigers_trapped = total_tigers - movable_tigers
        
        # Bucketize the features
        goats_captured_bucket = min(5, goats_captured)
        tigers_trapped_bucket = min(4, tigers_trapped)
        
        # Tiger mobility buckets
        if tiger_mobility <= 4:
            tiger_mobility_bucket = "Low:0-4"
        elif tiger_mobility <= 9:
            tiger_mobility_bucket = "Med:5-9"
        else:
            tiger_mobility_bucket = "High:10+"
        
        # Goat mobility buckets
        if goat_mobility <= 10:
            goat_mobility_bucket = "Low:0-10"
        elif goat_mobility <= 20:
            goat_mobility_bucket = "Med:11-20"
        else:
            goat_mobility_bucket = "High:21+"
        
        # Immediate captures buckets
        if immediate_captures == 0:
            captures_bucket = 0
        elif immediate_captures == 1:
            captures_bucket = 1
        else:
            captures_bucket = "2+"
        
        # Setup threats (threatened goats minus immediate captures)
        setup_threats = threatened_goats_count - immediate_captures
        if setup_threats == 0:
            threats_bucket = "None:0"
        elif setup_threats <= 2:
            threats_bucket = "Low:1-2"
        else:
            threats_bucket = "High:3+"
        
        # Tiger position score buckets
        if tiger_position_score < 0.3:
            position_bucket = "Low:0-0.3"
        elif tiger_position_score < 0.7:
            position_bucket = "Med:0.3-0.7"
        else:
            position_bucket = "High:0.7-1.0"
        
        # Tiger spacing score buckets
        if tiger_spacing_score < 0.3:
            spacing_bucket = "Low:0-0.3"
        elif tiger_spacing_score < 0.7:
            spacing_bucket = "Med:0.3-0.7"
        else:
            spacing_bucket = "High:0.7-1.0"
        
        # Goat edge score buckets
        if goat_edge_score < 0.4:
            edge_bucket = "Low:0-0.4"
        elif goat_edge_score < 0.7:
            edge_bucket = "Med:0.4-0.7"
        else:
            edge_bucket = "High:0.7-1.0"
        
        # Closed spaces buckets
        if total_closed_spaces == 0:
            closed_bucket = "None:0"
        elif total_closed_spaces <= 5:
            closed_bucket = "Low:1-5"
        else:
            closed_bucket = "High:6+"
        
        # Goats on board buckets
        if goats_on_board <= 8:
            goats_bucket = "Few:0-8"
        elif goats_on_board <= 15:
            goats_bucket = "Mid:9-15"
        else:
            goats_bucket = "Many:16-20"
        
        # Game phase
        phase = state.phase
        
        # Return the state features as a tuple for hash-ability
        return (
            goats_captured_bucket,
            tigers_trapped_bucket,
            tiger_mobility_bucket,
            goat_mobility_bucket,
            captures_bucket,
            threats_bucket,
            position_bucket,
            spacing_bucket,
            edge_bucket,
            closed_bucket,
            goats_bucket,
            phase
        )
    
    def _get_reward(self, prev_state: GameState, action: str, next_state: GameState) -> float:
        """Calculate rewards for goat actions"""
        # Check for terminal states first
        winner = next_state.get_winner()
        if winner == "GOAT":
            return 100.0
        elif winner == "TIGER":
            return -100.0
        
        # Base reward (small penalty per move to encourage shorter solutions)
        reward = -0.1
        
        # Penalty for being captured
        if next_state.goats_captured > prev_state.goats_captured:
            reward -= 25.0
        
        # Reward for trapping tigers
        prev_movable = _count_movable_tigers(get_all_possible_moves(prev_state.board, "MOVEMENT", "TIGER"))
        next_movable = _count_movable_tigers(get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER"))
        newly_trapped = prev_movable - next_movable
        if newly_trapped > 0:
            reward += 15.0 * newly_trapped
        
        # Reward for successful blocking
        if action == "Block_Immediate_Capture":
            reward += 15.0
        
        # Reward for successful escaping
        if action == "Escape_Threat":
            reward += 10.0
        
        # Reward for edge position improvement
        old_edge_score = _calculate_goat_edge_preference(prev_state)
        new_edge_score = _calculate_goat_edge_preference(next_state)
        edge_improvement = new_edge_score - old_edge_score
        if edge_improvement > 0:
            reward += 5.0 * edge_improvement
        
        # Reward for reducing tiger mobility
        prev_tiger_moves = len(get_all_possible_moves(prev_state.board, "MOVEMENT", "TIGER"))
        next_tiger_moves = len(get_all_possible_moves(next_state.board, "MOVEMENT", "TIGER"))
        mobility_reduction = prev_tiger_moves - next_tiger_moves
        if mobility_reduction > 0:
            reward += 2.0 * mobility_reduction
        
        return reward
    
    def update_q_table(self, state: GameState, action: str, reward: float, next_state: GameState):
        """Update Q-table and visit counts using adaptive learning rate"""
        # Get state features
        s = self._get_state_features(state)
        s_next = self._get_state_features(next_state)
        
        # Initialize Q-values if needed
        if s not in self.q_table:
            self.q_table[s] = {}
        if s not in self.visit_counts:
            self.visit_counts[s] = {}
        
        # Initialize action if needed
        if action not in self.q_table[s]:
            self.q_table[s][action] = 0.0
        if action not in self.visit_counts[s]:
            self.visit_counts[s][action] = 0
        
        # Increment visit count
        self.visit_counts[s][action] += 1
        
        # Calculate adaptive learning rate
        visit_count = self.visit_counts[s][action]
        alpha = 1.0 / (1.0 + visit_count)  # More stable adaptive learning rate
        
        # Get current Q-value
        current_q = self.q_table[s][action]
        
        # Get maximum Q-value for next state
        next_q_values = self.q_table.get(s_next, {})
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Update Q-value using Bellman equation with adaptive learning rate
        self.q_table[s][action] = current_q + alpha * (reward + self.discount_factor * max_next_q - current_q)


class QLearningAgent:
    """
    Main Q-learning agent class for Bagh Chal.
    Manages tiger and goat agents, training, and persistence.
    """
    
    def __init__(self, discount_factor=0.95, initial_exploration_rate=1.0, 
                 min_exploration_rate=0.05, exploration_decay=0.99, seed=None, 
                 auto_load=True, tables_path="backend/simulation_results/q_tables"):
        # Initialize configuration
        self.discount_factor = discount_factor
        self.initial_exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize role-specific agents
        self.tiger_agent = TigerAgent(discount_factor=discount_factor)
        self.goat_agent = GoatAgent(discount_factor=discount_factor)
        
        # Initialize random generator for reproducibility
        self.rng = random.Random(seed)
        
        # Tables path for auto-loading
        self.tables_path = tables_path
        
        # Auto-load tables if requested and they exist
        if auto_load:
            self._try_auto_load_tables()
    
    def _try_auto_load_tables(self):
        """Attempt to auto-load tables from the default location"""
        try:
            # Check for final tables first
            tiger_q = f"{self.tables_path}/tiger_q_final.json"
            tiger_v = f"{self.tables_path}/tiger_v_final.json"
            goat_q = f"{self.tables_path}/goat_q_final.json"
            goat_v = f"{self.tables_path}/goat_v_final.json"
            
            # If all final tables exist, load them
            if all(os.path.exists(f) for f in [tiger_q, tiger_v, goat_q, goat_v]):
                self.load_tables(tiger_q, tiger_v, goat_q, goat_v)
                print(f"Loaded final Q-tables from {self.tables_path}")
                return
            
            # Otherwise, look for metadata to find latest tables
            metadata_path = os.path.join(self.tables_path, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get the latest episode from metadata
                    latest_episode = metadata.get("completed_episodes", 0)
                    if latest_episode > 0:
                        # Create paths to latest tables
                        tiger_q = f"{self.tables_path}/tiger_q_{latest_episode}.json"
                        tiger_v = f"{self.tables_path}/tiger_v_{latest_episode}.json"
                        goat_q = f"{self.tables_path}/goat_q_{latest_episode}.json"
                        goat_v = f"{self.tables_path}/goat_v_{latest_episode}.json"
                        
                        # Check if the tables exist
                        if all(os.path.exists(f) for f in [tiger_q, tiger_v, goat_q, goat_v]):
                            self.load_tables(tiger_q, tiger_v, goat_q, goat_v)
                            print(f"Loaded Q-tables from episode {latest_episode}")
                            return
                except Exception as e:
                    print(f"Warning: Failed to load metadata: {e}")
            
            # If we get here, no tables were found
            print("No existing Q-tables found. Starting with new tables.")
        except Exception as e:
            print(f"Warning: Auto-loading tables failed: {e}")
    
    def get_move(self, state: GameState) -> Dict:
        """
        Interface method for getting a move during gameplay.
        Similar to minimax_agent.get_move().
        
        Args:
            state: Current game state
            
        Returns:
            A concrete move to apply
        """
        # Delegate to the appropriate agent based on whose turn it is
        if state.turn == "TIGER":
            return self.tiger_agent.get_move(state, epsilon=0.0)
        else:  # GOAT
            return self.goat_agent.get_move(state, epsilon=0.0)
    
    def load_from_default_path(self):
        """Convenience method to load tables from the default path"""
        self._try_auto_load_tables()
        
    def train_self_play(self, episodes: int, save_interval=100, save_path=None, max_time_seconds=None):
        """
        Train both agents through self-play.
        
        Args:
            episodes: Number of games to play
            save_interval: How often to save Q-tables (in episodes)
            save_path: Directory to save Q-tables
            max_time_seconds: Maximum training time in seconds (None for unlimited)
        """
        # If save_path not specified, use default path
        if save_path is None:
            save_path = self.tables_path
            
        # Rest of the method remains the same
        epsilon = self.initial_exploration_rate
        start_time = time.time()
        
        for episode in range(1, episodes + 1):
            # Check for time limit
            if max_time_seconds and time.time() - start_time > max_time_seconds:
                print(f"Training stopped due to time limit after {episode-1} episodes")
                break
                
            state = GameState()  # Start with new game
            
            # Track game statistics for reporting
            game_length = 0
            winner = None
            
            # Play until game end
            while not state.is_terminal():
                game_length += 1
                
                if state.turn == "TIGER":
                    # Tiger's turn
                    prev_state = state.clone()
                    abstract_action = self.tiger_agent._choose_action(state, epsilon)
                    if abstract_action is None:
                        # No valid actions, game should end
                        break
                        
                    concrete_move = self.tiger_agent._get_concrete_move(state, abstract_action)
                    if concrete_move is None:
                        # No valid moves, game should end
                        break
                    
                    # Apply move
                    state.apply_move(concrete_move)
                    
                    # Get immediate reward and update
                    reward = self.tiger_agent._get_reward(prev_state, abstract_action, state)
                    self.tiger_agent.update_q_table(prev_state, abstract_action, reward, state)
                    
                else:  # GOAT's turn
                    # Goat's turn
                    prev_state = state.clone()
                    abstract_action = self.goat_agent._choose_action(state, epsilon)
                    if abstract_action is None:
                        # No valid actions, game should end
                        break
                        
                    concrete_move = self.goat_agent._get_concrete_move(state, abstract_action)
                    if concrete_move is None:
                        # No valid moves, game should end
                        break
                    
                    # Apply move
                    state.apply_move(concrete_move)
                    
                    # Get immediate reward and update
                    reward = self.goat_agent._get_reward(prev_state, abstract_action, state)
                    self.goat_agent.update_q_table(prev_state, abstract_action, reward, state)
            
            # Record winner
            winner = state.get_winner()
            
            # Report progress periodically
            if episode % 10 == 0 or episode == 1:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.3f}, Game Length: {game_length}, Winner: {winner}, Elapsed: {elapsed_time:.1f}s")
            
            # Decay epsilon
            epsilon = max(self.min_exploration_rate, epsilon * self.exploration_decay)
            
            # Save periodically
            if save_path and episode % save_interval == 0:
                self.save_tables(
                    f"{save_path}/tiger_q_{episode}.json",
                    f"{save_path}/tiger_v_{episode}.json",
                    f"{save_path}/goat_q_{episode}.json",
                    f"{save_path}/goat_v_{episode}.json"
                )
        
        # Final save
        if save_path:
            self.save_tables(
                f"{save_path}/tiger_q_final.json",
                f"{save_path}/tiger_v_final.json",
                f"{save_path}/goat_q_final.json",
                f"{save_path}/goat_v_final.json"
            )
    
    def train_with_coach(self, episodes: int, coach_type: str, role_to_train: str, 
                         save_interval=100, save_path=None, coach_params=None, max_time_seconds=None):
        """
        Train one agent against a coach agent.
        
        Args:
            episodes: Number of games to play
            coach_type: "minimax" or "mcts"
            role_to_train: "TIGER" or "GOAT"
            save_interval: How often to save Q-tables (in episodes)
            save_path: Directory to save Q-tables
            coach_params: Parameters for the coach agent
            max_time_seconds: Maximum training time in seconds (None for unlimited)
        """
        # If save_path not specified, use default path
        if save_path is None:
            save_path = self.tables_path
            
        # Rest of the method remains the same
        epsilon = self.initial_exploration_rate
        start_time = time.time()
        
        # Initialize coach agent
        coach = self._create_coach_agent(coach_type, coach_params)
        if coach is None:
            print(f"Failed to create coach agent of type {coach_type}")
            return
        
        for episode in range(1, episodes + 1):
            # Check for time limit
            if max_time_seconds and time.time() - start_time > max_time_seconds:
                print(f"Training stopped due to time limit after {episode-1} episodes")
                break
                
            state = GameState()  # Start with new game
            
            # Track game statistics for reporting
            game_length = 0
            winner = None
            
            # Play until game end
            while not state.is_terminal():
                game_length += 1
                
                # Determine if the current player is the learner or the coach
                is_learner_turn = (state.turn == role_to_train)
                
                if is_learner_turn:
                    # Learner's turn
                    prev_state = state.clone()
                    
                    # Choose an action with exploration
                    if role_to_train == "TIGER":
                        abstract_action = self.tiger_agent._choose_action(state, epsilon)
                        if abstract_action is None:
                            break
                        concrete_move = self.tiger_agent._get_concrete_move(state, abstract_action)
                    else:  # GOAT
                        abstract_action = self.goat_agent._choose_action(state, epsilon)
                        if abstract_action is None:
                            break
                        concrete_move = self.goat_agent._get_concrete_move(state, abstract_action)
                        
                    if concrete_move is None:
                        break
                        
                    # Apply move
                    state.apply_move(concrete_move)
                    
                    # Get reward and update Q-table
                    if role_to_train == "TIGER":
                        reward = self.tiger_agent._get_reward(prev_state, abstract_action, state)
                        self.tiger_agent.update_q_table(prev_state, abstract_action, reward, state)
                    else:  # GOAT
                        reward = self.goat_agent._get_reward(prev_state, abstract_action, state)
                        self.goat_agent.update_q_table(prev_state, abstract_action, reward, state)
                    
                else:
                    # Coach's turn
                    coach_move = coach.get_move(state)
                    if coach_move is None:
                        break
                    
                    # Apply coach's move
                    state.apply_move(coach_move)
            
            # Record winner
            winner = state.get_winner()
            
            # Report progress periodically
            if episode % 10 == 0 or episode == 1:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.3f}, Game Length: {game_length}, Winner: {winner}, Elapsed: {elapsed_time:.1f}s")
            
            # Decay epsilon
            epsilon = max(self.min_exploration_rate, epsilon * self.exploration_decay)
            
            # Save periodically
            if save_path and episode % save_interval == 0:
                if role_to_train == "TIGER":
                    self.save_tables(
                        f"{save_path}/tiger_q_{episode}.json",
                        f"{save_path}/tiger_v_{episode}.json",
                        None,
                        None
                    )
                else:  # GOAT
                    self.save_tables(
                        None,
                        None,
                        f"{save_path}/goat_q_{episode}.json",
                        f"{save_path}/goat_v_{episode}.json"
                    )
        
        # Final save
        if save_path:
            if role_to_train == "TIGER":
                self.save_tables(
                    f"{save_path}/tiger_q_final.json",
                    f"{save_path}/tiger_v_final.json",
                    None,
                    None
                )
            else:  # GOAT
                self.save_tables(
                    None,
                    None,
                    f"{save_path}/goat_q_final.json",
                    f"{save_path}/goat_v_final.json"
                )
    
    def _create_coach_agent(self, coach_type: str, coach_params=None):
        """Create a coach agent of the specified type with the given parameters"""
        if coach_params is None:
            coach_params = {}
            
        if coach_type.lower() == "minimax":
            from models.minimax_agent import MinimaxAgent
            # Default parameters if not specified
            max_depth = coach_params.get("max_depth", 3)
            randomize_equal_moves = coach_params.get("randomize_equal_moves", False)
            return MinimaxAgent(max_depth=max_depth, randomize_equal_moves=randomize_equal_moves)
            
        elif coach_type.lower() == "mcts":
            from models.mcts_agent import MCTSAgent
            # Default parameters if not specified
            iterations = coach_params.get("iterations", 1000)
            exploration_weight = coach_params.get("exploration_weight", 1.0)
            return MCTSAgent(iterations=iterations, exploration_weight=exploration_weight)
            
        else:
            print(f"Unknown coach type: {coach_type}")
            return None
    
    def save_tables(self, tiger_q_filepath, tiger_v_filepath, goat_q_filepath, goat_v_filepath):
        """Save Q-tables and visit counts to disk"""
        try:
            # Helper function to serialize q-table or visit counts
            def serialize_table(table):
                # Convert tuples to strings for JSON serialization
                serialized = {}
                for state, actions in table.items():
                    # Convert state tuple to string
                    state_key = str(state)
                    serialized[state_key] = actions
                return serialized
            
            # Ensure directories exist
            if tiger_q_filepath:
                os.makedirs(os.path.dirname(tiger_q_filepath), exist_ok=True)
            if tiger_v_filepath:
                os.makedirs(os.path.dirname(tiger_v_filepath), exist_ok=True)
            if goat_q_filepath:
                os.makedirs(os.path.dirname(goat_q_filepath), exist_ok=True)
            if goat_v_filepath:
                os.makedirs(os.path.dirname(goat_v_filepath), exist_ok=True)
                
            # Save tiger tables if paths are provided
            if tiger_q_filepath:
                with open(tiger_q_filepath, 'w') as f:
                    json.dump(serialize_table(self.tiger_agent.q_table), f)
                print(f"Saved tiger Q-table to {tiger_q_filepath}")
                
            if tiger_v_filepath:
                with open(tiger_v_filepath, 'w') as f:
                    json.dump(serialize_table(self.tiger_agent.visit_counts), f)
                print(f"Saved tiger visit counts to {tiger_v_filepath}")
            
            # Save goat tables if paths are provided
            if goat_q_filepath:
                with open(goat_q_filepath, 'w') as f:
                    json.dump(serialize_table(self.goat_agent.q_table), f)
                print(f"Saved goat Q-table to {goat_q_filepath}")
                
            if goat_v_filepath:
                with open(goat_v_filepath, 'w') as f:
                    json.dump(serialize_table(self.goat_agent.visit_counts), f)
                print(f"Saved goat visit counts to {goat_v_filepath}")
                
        except Exception as e:
            print(f"Error saving tables: {e}")
    
    def load_tables(self, tiger_q_filepath, tiger_v_filepath, goat_q_filepath, goat_v_filepath):
        """Load Q-tables and visit counts from disk"""
        try:
            # Helper function to deserialize q-table or visit counts
            def deserialize_table(serialized):
                # Convert string keys back to tuples
                deserialized = {}
                for state_key, actions in serialized.items():
                    # Convert string representation of tuple back to actual tuple
                    # Using eval is safe here since we're controlling the input format
                    state_tuple = eval(state_key)
                    deserialized[state_tuple] = actions
                return deserialized
            
            # Load tiger tables if paths are provided
            if tiger_q_filepath and os.path.exists(tiger_q_filepath):
                with open(tiger_q_filepath, 'r') as f:
                    self.tiger_agent.q_table = deserialize_table(json.load(f))
                print(f"Loaded tiger Q-table from {tiger_q_filepath}")
                
            if tiger_v_filepath and os.path.exists(tiger_v_filepath):
                with open(tiger_v_filepath, 'r') as f:
                    self.tiger_agent.visit_counts = deserialize_table(json.load(f))
                print(f"Loaded tiger visit counts from {tiger_v_filepath}")
            
            # Load goat tables if paths are provided
            if goat_q_filepath and os.path.exists(goat_q_filepath):
                with open(goat_q_filepath, 'r') as f:
                    self.goat_agent.q_table = deserialize_table(json.load(f))
                print(f"Loaded goat Q-table from {goat_q_filepath}")
                
            if goat_v_filepath and os.path.exists(goat_v_filepath):
                with open(goat_v_filepath, 'r') as f:
                    self.goat_agent.visit_counts = deserialize_table(json.load(f))
                print(f"Loaded goat visit counts from {goat_v_filepath}")
                
        except Exception as e:
            print(f"Error loading tables: {e}") 