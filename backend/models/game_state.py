from typing import List, Dict, Optional, Tuple
from copy import deepcopy
from game_logic import get_possible_moves, get_all_possible_moves, get_threatened_nodes

class GameState:
    """
    A class representing the state of a Bagh Chal game.
    This class is designed to be used by various game-playing algorithms (Minimax, MCTS, etc.)
    """
    
    TOTAL_GOATS = 20
    WINNING_CAPTURES = 5
    BOARD_SIZE = 5

    def __init__(self):
        # Initialize empty board
        self.board = [[None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        
        # Place tigers in corners
        corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for x, y in corners:
            self.board[y][x] = {"type": "TIGER"}
        
        # Game state
        self.goats_placed = 0
        self.goats_captured = 0
        self.turn = "GOAT"  # "GOAT" or "TIGER"
        self.phase = "PLACEMENT"  # "PLACEMENT" or "MOVEMENT"
        self.game_status = "PLAYING"  # "PLAYING", "TIGERS_WIN", or "GOATS_WIN"

    @staticmethod
    def _moves_equal(move1: Dict, move2: Dict) -> bool:
        """Compare two moves for equality."""
        if move1["type"] != move2["type"]:
            return False
            
        if move1["type"] == "placement":
            return move1["x"] == move2["x"] and move1["y"] == move2["y"]
        
        # For movement moves
        if (move1["from"]["x"] != move2["from"]["x"] or 
            move1["from"]["y"] != move2["from"]["y"] or
            move1["to"]["x"] != move2["to"]["x"] or
            move1["to"]["y"] != move2["to"]["y"]):
            return False
            
        # Check capture if present in either move
        cap1 = move1.get("capture")
        cap2 = move2.get("capture")
        if cap1 and cap2:
            return cap1["x"] == cap2["x"] and cap1["y"] == cap2["y"]
        return cap1 is None and cap2 is None

    def clone(self) -> 'GameState':
        """Create a deep copy of the current game state."""
        new_state = GameState()
        new_state.board = deepcopy(self.board)
        new_state.goats_placed = self.goats_placed
        new_state.goats_captured = self.goats_captured
        new_state.turn = self.turn
        new_state.phase = self.phase
        new_state.game_status = self.game_status
        return new_state

    def get_valid_moves(self) -> List[Dict]:
        """Get all valid moves for the current player."""
        if self.is_terminal():
            return []
        return get_all_possible_moves(self.board, self.phase, self.turn)

    def apply_move(self, move: Dict) -> None:
        """Apply a move to the current game state."""
        if self.is_terminal():
            raise ValueError("Game is already over")

        # Type checking
        required_keys = {"type"}
        if move["type"] == "placement":
            required_keys.update({"x", "y"})
        else:
            required_keys.update({"from", "to"})
            
        if not all(key in move for key in required_keys):
            raise ValueError(f"Move missing required keys: {required_keys}")

        def validate_coords(x: int, y: int) -> None:
            """Validate coordinates are within board bounds."""
            if not (0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE):
                raise ValueError(f"Coordinates ({x}, {y}) out of bounds")

        # Validate move
        valid_moves = self.get_valid_moves()
        if not any(self._moves_equal(move, valid_move) for valid_move in valid_moves):
            raise ValueError("Invalid move")

        if move["type"] == "placement":
            if self.phase != "PLACEMENT" or self.turn != "GOAT":
                raise ValueError("Placement move not allowed in current state")
            validate_coords(move["x"], move["y"])
            # Place a goat
            self.board[move["y"]][move["x"]] = {"type": "GOAT"}
            self.goats_placed += 1
            if self.goats_placed >= self.TOTAL_GOATS:
                self.phase = "MOVEMENT"
        else:
            # Type checking for movement
            if not isinstance(move["from"], dict) or not isinstance(move["to"], dict):
                raise ValueError("Invalid movement move format")
            
            # Move a piece
            from_x, from_y = move["from"]["x"], move["from"]["y"]
            to_x, to_y = move["to"]["x"], move["to"]["y"]
            
            # Validate coordinates
            validate_coords(from_x, from_y)
            validate_coords(to_x, to_y)
            
            # Validate piece ownership
            piece = self.board[from_y][from_x]
            if not piece or piece["type"] != self.turn:
                raise ValueError("Cannot move opponent's piece or empty space")
            
            # Move the piece
            self.board[to_y][to_x] = self.board[from_y][from_x]
            self.board[from_y][from_x] = None
            
            # Handle capture
            if move.get("capture"):
                cap_x, cap_y = move["capture"]["x"], move["capture"]["y"]
                validate_coords(cap_x, cap_y)
                captured_piece = self.board[cap_y][cap_x]
                if not captured_piece or captured_piece["type"] != "GOAT":
                    raise ValueError("Invalid capture: no goat at capture position")
                self.board[cap_y][cap_x] = None
                self.goats_captured += 1

        # Switch turns
        self.turn = "TIGER" if self.turn == "GOAT" else "GOAT"
        
        # Check for game end
        self._update_game_status()

    def _update_game_status(self) -> None:
        """Update the game status based on the current state."""
        # Check if tigers won (5 goats captured)
        if self.goats_captured >= self.WINNING_CAPTURES:
            self.game_status = "TIGERS_WIN"
            return

        # Only check for no-moves win condition in movement phase
        if self.phase == "MOVEMENT":
            # Get valid moves for current player
            valid_moves = self.get_valid_moves()
            
            if not valid_moves:
                # If current player has no moves, they lose
                self.game_status = "TIGERS_WIN" if self.turn == "GOAT" else "GOATS_WIN"
                return

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.game_status != "PLAYING"

    def get_winner(self) -> Optional[str]:
        """Get the winner of the game if it has ended."""
        if self.game_status == "PLAYING":
            return None
        return "TIGER" if self.game_status == "TIGERS_WIN" else "GOAT"

    def get_threatened_nodes(self) -> List[Tuple[int, int]]:
        """
        Get all board positions where if a goat is placed, it could be immediately captured by a tiger.
        This is an efficient implementation that doesn't clone the state or perform unnecessary move generation.
        
        Returns:
            A list of (x, y) coordinates representing threatened positions.
        """
        return get_threatened_nodes(self.board) 