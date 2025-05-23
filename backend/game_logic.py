def is_in_bounds(x, y):
    """Check if coordinates are within the 5x5 board bounds."""
    return 0 <= x < 5 and 0 <= y < 5

def is_outer_layer(x, y):
    """Check if a position is on the outer layer of the board."""
    return x == 0 or y == 0 or x == 4 or y == 4

def is_second_layer(x, y):
    """Check if a position is on the second layer of the board."""
    return x == 1 or y == 1 or x == 3 or y == 3

def is_valid_connection(from_x, from_y, to_x, to_y):
    """
    Check if there is a valid connection between two positions on the board.
    """
    # Orthogonal moves are always valid if adjacent
    if abs(from_x - to_x) + abs(from_y - to_y) == 1:
        return True

    # Diagonal moves need special handling
    if abs(from_x - to_x) == 1 and abs(from_y - to_y) == 1:
        # No diagonal moves for second and fourth nodes on outer edges
        if is_outer_layer(from_x, from_y):
            is_second_or_fourth_node = (
                ((from_x == 0 or from_x == 4) and (from_y == 1 or from_y == 3)) or
                ((from_y == 0 or from_y == 4) and (from_x == 1 or from_x == 3))
            )
            if is_second_or_fourth_node:
                return False

        # No diagonal moves for middle nodes in second layer
        if is_second_layer(from_x, from_y):
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

def get_possible_moves(x, y, board):
    moves = []
    piece = board[y][x]
    if not piece:
        return moves

    # Check regular moves
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            new_x = x + dx
            new_y = y + dy

            if (
                is_in_bounds(new_x, new_y) and
                board[new_y][new_x] is None and
                is_valid_connection(x, y, new_x, new_y)
            ):
                moves.append({"x": new_x, "y": new_y, "type": "regular"})

    # Add capture moves for tigers
    if piece["type"] == "TIGER":
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            mid_x = x + dx
            mid_y = y + dy
            jump_x = x + dx * 2
            jump_y = y + dy * 2

            # Check if the jump is valid:
            # 1. All points must be in bounds
            # 2. Middle point must have a goat
            # 3. Destination must be empty
            # 4. Must have valid connection from tiger to goat
            # 5. Must have valid connection from goat to destination
            if (
                is_in_bounds(jump_x, jump_y) and
                is_in_bounds(mid_x, mid_y) and
                board[mid_y][mid_x] is not None and
                board[mid_y][mid_x]["type"] == "GOAT" and
                board[jump_y][jump_x] is None and
                is_valid_connection(x, y, mid_x, mid_y) and  # Tiger to Goat connection
                is_valid_connection(mid_x, mid_y, jump_x, jump_y)  # Goat to Destination connection
            ):
                moves.append({
                    "x": jump_x,
                    "y": jump_y,
                    "type": "capture",
                    "capturedPiece": {"x": mid_x, "y": mid_y}
                })

    return moves

def get_all_possible_moves(board, phase, piece_type):
    """Get all possible moves for the current player."""
    moves = []
    
    if phase == "PLACEMENT" and piece_type == "GOAT":
        # Get all empty spaces for goat placement
        for y in range(len(board)):
            for x in range(len(board[y])):
                if board[y][x] is None:
                    moves.append({"type": "placement", "x": x, "y": y})
    else:
        # Get all possible moves for each piece
        for y in range(len(board)):
            for x in range(len(board[y])):
                if board[y][x] and board[y][x]["type"] == piece_type:
                    piece_moves = get_possible_moves(x, y, board)
                    for move in piece_moves:
                        # Convert the move to match frontend structure
                        movement = {
                            "type": "movement",
                            "from": {"x": x, "y": y},
                            "to": {"x": move["x"], "y": move["y"]}
                        }
                        if move.get("capturedPiece"):
                            movement["capture"] = move["capturedPiece"]
                        moves.append(movement)
    
    return moves

def get_threatened_nodes(board):
    """
    Get all potential tiger capture paths, regardless of whether positions are currently occupied.
    
    Args:
        board: The current game board
        
    Returns:
        A list of tuples (x, y, landing_x, landing_y) where:
        - (x, y) are coordinates of a position that could be involved in a capture
        - (landing_x, landing_y) are coordinates where tiger would land after capturing
        
    Note:
        This includes all possible capture scenarios:
        1. Direct threats (empty goat position + empty landing)
        2. Blocking opportunities (goat on hot square + empty landing)
        3. Escape opportunities (goat on hot square)
    """
    threatened_nodes = []
    
    # Find all tigers on the board
    tiger_positions = []
    for y in range(5):
        for x in range(5):
            if board[y][x] is not None and board[y][x]["type"] == "TIGER":
                tiger_positions.append((x, y))
    
    # For each tiger, check potential capture moves
    for tiger_x, tiger_y in tiger_positions:
        # Check all 8 possible directions
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            # Calculate potential goat position (adjacent to tiger)
            goat_x = tiger_x + dx
            goat_y = tiger_y + dy
            
            # Calculate landing position (two steps from tiger)
            landing_x = tiger_x + (2 * dx)
            landing_y = tiger_y + (2 * dy)
            
            # Check if both positions are in bounds
            if not (is_in_bounds(goat_x, goat_y) and is_in_bounds(landing_x, landing_y)):
                continue
            
            # Note: We are NOT checking if the goat position is empty anymore
            # This allows finding potential capture paths even when positions are occupied
                
            # Check connectivity from tiger to goat
            if not is_valid_connection(tiger_x, tiger_y, goat_x, goat_y):
                continue
                
            # Check connectivity from goat to landing
            if not is_valid_connection(goat_x, goat_y, landing_x, landing_y):
                continue
            
            # Add to threatened nodes with landing coordinates
            threatened_nodes.append((goat_x, goat_y, landing_x, landing_y))
    
    return threatened_nodes

def check_game_end(board, goats_captured):
    """Check if the game has ended and return the winner if any."""
    # Check if tigers won (5 goats captured)
    if goats_captured >= 5:
        return "TIGERS_WIN"

    # Check if tigers have no legal moves (works in both phases)
    tiger_has_move = False
    for y in range(len(board)):
        for x in range(len(board[y])):
            if board[y][x] and board[y][x]["type"] == "TIGER":
                moves = get_possible_moves(x, y, board)
                if moves:
                    tiger_has_move = True
                    break
        if tiger_has_move:
            break

    if not tiger_has_move:
        return "GOATS_WIN"

    return "PLAYING" 