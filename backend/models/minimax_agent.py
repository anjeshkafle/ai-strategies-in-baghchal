from typing import List, Optional, Dict, Union, TypedDict
import time
from game_logic import get_all_possible_moves
from models.game_state import GameState

class PlacementMove(TypedDict):
    type: str  # "placement"
    x: int
    y: int

class MovementMove(TypedDict):
    type: str  # "movement"
    from_: Dict[str, int]
    to: Dict[str, int]
    capture: Optional[Dict[str, int]]

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for the Bagh Chal game.
    Can be configured with either a maximum depth or maximum time limit.
    """
    
    def __init__(self, max_depth: Optional[int] = 4, max_time: Optional[float] = None):
        """
        Initialize the minimax agent.
        
        Args:
            max_depth: Maximum depth to search in the game tree. Default is 4.
            max_time: Maximum time in seconds to search. If set, overrides max_depth.
        """
        self.max_depth = max_depth
        self.max_time = max_time
        self.start_time = 0
        self.timeout = False
        self.nodes_explored = 0
        
        # Position weights for the board (center positions are more valuable)
        self.position_weights = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ]
    
    def get_move(self, board: List[List[Optional[Dict]]], phase: str, agent: str) -> Union[PlacementMove, MovementMove]:
        """
        Get the best move for the current state using minimax with alpha-beta pruning.
        
        Args:
            board: The current game board
            phase: Current game phase ("PLACEMENT" or "MOVEMENT")
            agent: The agent's piece type ("TIGER" or "GOAT")
            
        Returns:
            The best move found
        """
        print(f"\nMinimaxAgent: Starting move search for {agent} in {phase} phase")
        
        # Create a game state from the board
        game_state = GameState()
        game_state.board = board
        game_state.phase = phase
        game_state.turn = agent.upper()
        
        # Count goats on board and captured
        goats_on_board = sum(1 for row in board for cell in row if cell and cell.get("type") == "GOAT")
        game_state.goats_placed = goats_on_board
        game_state.goats_captured = 20 - goats_on_board if phase == "MOVEMENT" else 0
        
        print(f"MinimaxAgent: Board state - {goats_on_board} goats on board, {game_state.goats_captured} captured")
        
        # Use minimax to select the best move
        try:
            move = self.select_move(game_state)
            print(f"MinimaxAgent: Selected move: {move}")
            return move
        except Exception as e:
            print(f"MinimaxAgent: Error selecting move - {str(e)}")
            # Return first valid move as fallback
            valid_moves = game_state.get_valid_moves()
            if not valid_moves:
                raise ValueError("No valid moves available")
            return valid_moves[0]
    
    def select_move(self, game_state) -> Dict:
        """Select the best move using minimax with alpha-beta pruning."""
        self.start_time = time.time()
        self.timeout = False
        self.nodes_explored = 0
        
        # Get all valid moves
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # If only one move is available, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Initialize best move tracking
        is_tiger = game_state.turn == "TIGER"
        best_move = None
        best_value = float('-inf') if is_tiger else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves before evaluation
        ordered_moves = self._order_moves(game_state, valid_moves)
        
        print(f"MinimaxAgent: Evaluating {len(ordered_moves)} possible moves")
        
        # Try each move and evaluate
        for i, move in enumerate(ordered_moves):
            # Clone the game state and apply the move
            new_state = game_state.clone()
            new_state.apply_move(move)
            
            # Get value from minimax
            value = self._minimax(new_state, self.max_depth - 1, alpha, beta, not is_tiger)
            
            print(f"MinimaxAgent: Move {i+1}/{len(ordered_moves)} evaluated with value {value}")
            
            if is_tiger:  # Tiger's turn (maximizing)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:  # Goat's turn (minimizing)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
            
            # Check for timeout
            if self.max_time and time.time() - self.start_time > self.max_time:
                print("MinimaxAgent: Search timed out")
                self.timeout = True
                break
        
        elapsed = time.time() - self.start_time
        print(f"MinimaxAgent: Search completed in {elapsed:.2f}s. Explored {self.nodes_explored} nodes ({self.nodes_explored/elapsed:.0f} nodes/s)")
        print(f"MinimaxAgent: Selected move with value {best_value}")
        
        return best_move or valid_moves[0]
    
    def _order_moves(self, state, moves: List[Dict]) -> List[Dict]:
        """
        Order moves to optimize alpha-beta pruning.
        Prioritizes:
        1. Capture moves
        2. Moves to/from central positions
        3. Moves that increase mobility
        """
        move_scores = []
        is_tiger = state.turn == "TIGER"
        
        for move in moves:
            score = 0
            
            # Prioritize captures highest (for tigers)
            if "capture" in move:
                score += 1000 if is_tiger else -1000
            
            # Score based on position value
            if move["type"] == "placement":
                weight = self.position_weights[move["y"]][move["x"]]
                score += weight * (10 if is_tiger else -10)
            elif move["type"] == "movement":
                to_weight = self.position_weights[move["to"]["y"]][move["to"]["x"]]
                from_weight = self.position_weights[move["from"]["y"]][move["from"]["x"]]
                weight_diff = to_weight - from_weight
                score += weight_diff * (10 if is_tiger else -10)
            
            # Quick mobility heuristic
            new_state = state.clone()
            new_state.apply_move(move)
            tiger_moves = len(get_all_possible_moves(new_state.board, new_state.phase, "TIGER"))
            goat_moves = len(get_all_possible_moves(new_state.board, new_state.phase, "GOAT"))
            
            # Add mobility score based on player
            if is_tiger:
                score += tiger_moves - goat_moves
            else:
                score += goat_moves - tiger_moves
            
            move_scores.append((score, move))
        
        # Sort moves by score (descending for tigers, ascending for goats)
        move_scores.sort(reverse=is_tiger, key=lambda x: x[0])
        return [move for _, move in move_scores]
    
    def _minimax(self, state, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning and quiescence search.
        
        Args:
            state: Current game state
            depth: Current depth in the search tree
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player (Tiger), False if minimizing (Goat)
        
        Returns:
            float: The evaluation score for the current state
        """
        self.nodes_explored += 1
        
        # Check for timeout first
        if self.max_time and time.time() - self.start_time > self.max_time:
            self.timeout = True
            return self._evaluate_state(state)
        
        # Check for terminal state
        if state.is_terminal():
            winner = state.get_winner()
            if winner == "TIGER":
                return float('inf') - depth  # Tigers win (prefer faster wins)
            else:
                return float('-inf') + depth  # Goats win (prefer faster wins)
            
        # If at depth limit, do a quiescence search for captures
        if depth == 0:
            # Get tiger moves once and cache them
            tiger_moves = get_all_possible_moves(state.board, state.phase, "TIGER")
            capture_moves = [move for move in tiger_moves if "capture" in move]
            
            # Only continue search if there are captures and we haven't gone too deep
            # Limit quiescence search to prevent excessive depth
            if capture_moves and depth > -2:  # Reduced max quiescence depth to 2
                if not maximizing:
                    # For goats, continue if there are threats against them
                    depth = 1
                else:
                    # For tigers, continue if they have capture opportunities
                    depth = 1
                
                # Continue with normal minimax since we've adjusted depth
                valid_moves = state.get_valid_moves()
                if not valid_moves:
                    return self._evaluate_state(state)
                
                # For quiescence, only consider capture moves for tigers
                if maximizing and state.phase == "MOVEMENT":
                    valid_moves = [move for move in valid_moves if "capture" in move]
                    if not valid_moves:  # If no capture moves, evaluate position
                        return self._evaluate_state(state)
                
                # Order moves before evaluation
                ordered_moves = self._order_moves(state, valid_moves)
                
                if maximizing:  # Tiger's turn
                    value = float('-inf')
                    for move in ordered_moves:
                        new_state = state.clone()
                        new_state.apply_move(move)
                        value = max(value, self._minimax(new_state, depth - 1, alpha, beta, False))
                        alpha = max(alpha, value)
                        if beta <= alpha or self.timeout:
                            break
                    return value
                else:  # Goat's turn
                    value = float('inf')
                    for move in ordered_moves:
                        new_state = state.clone()
                        new_state.apply_move(move)
                        value = min(value, self._minimax(new_state, depth - 1, alpha, beta, True))
                        beta = min(beta, value)
                        if beta <= alpha or self.timeout:
                            break
                    return value
            
            # If no captures or too deep, evaluate position
            return self._evaluate_state(state)
        
        # Normal minimax search for non-zero depth
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            return self._evaluate_state(state)
        
        # Order moves before evaluation
        ordered_moves = self._order_moves(state, valid_moves)
        
        if maximizing:  # Tiger's turn
            value = float('-inf')
            for move in ordered_moves:
                new_state = state.clone()
                new_state.apply_move(move)
                value = max(value, self._minimax(new_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha or self.timeout:
                    break
            return value
        else:  # Goat's turn
            value = float('inf')
            for move in ordered_moves:
                new_state = state.clone()
                new_state.apply_move(move)
                value = min(value, self._minimax(new_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha or self.timeout:
                    break
            return value
    
    def _evaluate_state(self, state) -> float:
        """
        Evaluate the current game state.
        Positive values favor tigers, negative values favor goats.
        
        The evaluation takes into account:
        1. Number of goats captured (major advantage for tigers)
        2. Immediate capture threats
        3. Number of available moves (mobility)
        4. Board position control
        5. Phase-specific considerations
        """
        # Base score from captures (each capture is worth 100 points)
        score = state.goats_captured * 100
        
        # Check for immediate capture threats (critical for goats to avoid)
        if state.phase == "MOVEMENT":
            tiger_moves = get_all_possible_moves(state.board, state.phase, "TIGER")
            capture_moves = [move for move in tiger_moves if "capture" in move]
            if capture_moves:
                # Much higher penalty for capture threats - should be higher than actual captures
                # since preventing captures is more important than the captures themselves
                score += 500 * len(capture_moves)
        elif state.phase == "PLACEMENT":
            # During placement, check if placing a goat would allow an immediate capture
            tiger_moves = get_all_possible_moves(state.board, "MOVEMENT", "TIGER")
            capture_moves = [move for move in tiger_moves if "capture" in move]
            if capture_moves:
                # Even higher penalty during placement phase since goats have more placement options
                score += 1000 * len(capture_moves)
        
        # Get mobility scores
        tiger_moves = len(get_all_possible_moves(state.board, state.phase, "TIGER"))
        goat_moves = len(get_all_possible_moves(state.board, state.phase, "GOAT"))
        
        # Weight mobility differently based on phase
        if state.phase == "PLACEMENT":
            # During placement, goat mobility is less important as they can place anywhere
            # But tiger mobility is still crucial
            score += (tiger_moves * 15 - goat_moves * 3)
        else:
            # During movement, mobility is crucial for both
            score += (tiger_moves - goat_moves) * 10
        
        # Add position-based scoring
        for y in range(5):
            for x in range(5):
                piece = state.board[y][x]
                if piece:
                    weight = self.position_weights[y][x]
                    if piece["type"] == "TIGER":
                        # Tigers value center positions more in movement phase
                        pos_multiplier = 7 if state.phase == "MOVEMENT" else 5
                        score += weight * pos_multiplier
                    else:
                        # Goats value center positions more in placement phase
                        # But not at the cost of safety
                        pos_multiplier = 4 if state.phase == "PLACEMENT" else 3
                        score -= weight * pos_multiplier
        
        return score 