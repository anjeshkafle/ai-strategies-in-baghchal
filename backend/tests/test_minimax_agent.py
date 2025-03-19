import sys
import os
# Add the parent directory to the path so that we can import from 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
import json

def convert_string_to_board(board_state):
    """Convert string representation to game board.
    Each string in the list represents a row from top to bottom.
    T = Tiger, G = Goat, _ = Empty
    
    Note: The internal board representation uses [y][x] indexing,
    so we need to swap coordinates when converting from visual representation.
    """
    board = [[None for _ in range(5)] for _ in range(5)]
    for row_idx, row in enumerate(board_state):
        for col_idx, cell in enumerate(row):
            if cell == 'T':
                board[row_idx][col_idx] = {"type": "TIGER"}  # Don't swap - GameState uses [row][col]
            elif cell == 'G':
                board[row_idx][col_idx] = {"type": "GOAT"}
            # _ represents empty cell, already None by default
    return board

def format_move(move):
    """Format a move object for display in the move column"""
    try:
        if isinstance(move, str):
            # Try to convert string representation to dict
            try:
                move_dict = eval(move)
                return format_move(move_dict)
            except:
                return f"STR:{move[:20]}"  # Shortened string representation
        
        if move['type'] == 'movement':
            from_x = move['from']['x']
            from_y = move['from']['y']
            to_x = move['to']['x']
            to_y = move['to']['y']
            capture = "C" if move.get('capture', False) else ""
            return f"({from_x},{from_y})→({to_x},{to_y}){capture}"
        elif move['type'] == 'placement':
            # Handle different formats of placement moves
            if 'to' in move:
                to_x = move['to']['x'] 
                to_y = move['to']['y']
            elif 'x' in move and 'y' in move:
                to_x = move['x']
                to_y = move['y']
            else:
                return f"P(?,?)"
            return f"P({to_x},{to_y})"
        return f"??{str(move)[:10]}"
    except Exception as e:
        return f"Error:{str(e)[:15]}"

def format_move_short(move):
    """Format a move object for more concise display"""
    try:
        if isinstance(move, str):
            # Try to convert string representation to dict
            try:
                move_dict = eval(move)
                return format_move_short(move_dict)
            except:
                return f"STR:{move[:10]}"  # Shortened string representation
        
        if move['type'] == 'movement':
            from_x = move['from']['x']
            from_y = move['from']['y']
            to_x = move['to']['x']
            to_y = move['to']['y']
            capture = "C" if move.get('capture', False) else ""
            return f"({from_x},{from_y})-({to_x},{to_y}){capture}"
        elif move['type'] == 'placement':
            # Handle different formats of placement moves
            if 'to' in move:
                to_x = move['to']['x'] 
                to_y = move['to']['y']
            elif 'x' in move and 'y' in move:
                to_x = move['x']
                to_y = move['y']
            else:
                return f"P(?,?)"
            return f"P({to_x},{to_y})"
        return f"?{str(move)[:5]}"
    except Exception as e:
        # For debugging
        return f"?{str(move)[:10]}"

def test_minimax_agent(depth=4):
    test_state_2 = [
      "_____",
      "_TGG_",
      "_____",
      "_____",
      "T__TT"
    ]

    test_state_3 = [
      "T___T",
      "_____",
      "___G_",
      "_____",
      "T___T"
    ]

    test_state_1 = [
        "TTT__",
        "_____",
        "_____",
        "____G",
        "____T"
    ]
    
    print("\n" + "="*50)
    print("TESTING MINIMAX AGENT")
    print("="*50)
    
    # Create game state
    game_state_1 = GameState()
    game_state_1.board = convert_string_to_board(test_state_1)
    game_state_1.goats_placed = 4  # Set to placement phase
    game_state_1.phase = "PLACEMENT"
    game_state_1.turn = "TIGER"  # Explicitly set turn
    
    # Create minimax agent with specified depth
    agent = MinimaxAgent(max_depth=depth)
    
    # Get best move from minimax agent
    best_move = agent.get_move(game_state_1)
    print(f"\nBest move: {format_move(best_move)}")
    print(f"Evaluation score: {agent.best_score}")
    
    # Display evaluation breakdown for all moves
    if hasattr(agent, 'all_move_evaluations') and agent.all_move_evaluations:
        print("\n=== Move Evaluations ===")
        headers = ["Move", "Score", "Tigers", "Goats", "Closed", "Depth", "Chosen", "Move Sequence"]
        print(f"{headers[0]:25} | {headers[1]:6} | {headers[2]:6} | {headers[3]:6} | {headers[4]:6} | {headers[5]:5} | {headers[6]:6} | {headers[7]}")
        print("-" * 100)
        
        for eval_data in sorted(agent.all_move_evaluations, 
                              key=lambda x: x['score'], 
                              reverse=(game_state_1.turn == "TIGER")):
            # Format the move for display
            move = eval_data['move']
            if move['type'] == 'placement':
                move_str = f"Place at ({move['to']['x']},{move['to']['y']})"
            else:  # movement
                move_str = f"Move({move['from']['x']},{move['from']['y']})-({move['to']['x']},{move['to']['y']})"
                if move.get('capture', False):
                    move_str += " [C]"
            
            # Get score and breakdown
            minimax_score = eval_data['score']
            is_best = "✓" if eval_data['is_best'] else ""
            
            # Get the leaf node evaluation components that determined this score
            if 'leaf_node' in eval_data:
                leaf = eval_data['leaf_node']
                tigers = leaf['movable_tigers']
                goats = leaf['goats_captured']
                closed = leaf['closed_spaces']
                depth = leaf.get('depth_penalty', 0)
                
                # Format the complete move sequence
                sequence = ""
                if 'move_sequence' in leaf and leaf['move_sequence']:
                    move_seq = leaf['move_sequence']
                    formatted_moves = []
                    
                    # Add sequence length
                    sequence_length = len(move_seq)
                    sequence = f"[{sequence_length} moves] "
                    
                    # Format each move with player prefix
                    for i, m in enumerate(move_seq):
                        player_prefix = "T:" if i % 2 == 0 else "G:"
                        
                        # For string representations, parse them to dictionaries
                        if isinstance(m, str):
                            try:
                                # Convert string representation to dictionary by evaluating it
                                m_dict = eval(m)
                                formatted_moves.append(f"{player_prefix}{format_move_short(m_dict)}")
                            except:
                                formatted_moves.append(f"{player_prefix}{m[:15]}...")
                        else:
                            formatted_moves.append(f"{player_prefix}{format_move_short(m)}")
                    
                    # Join all moves with arrows
                    sequence += " → ".join(formatted_moves)
                
                # Print the evaluation with each component
                print(f"{move_str:25} | {minimax_score:6} | {tigers:6} | {goats:6} | {closed:6} | {depth:5} | {is_best:^6} | {sequence}")
            else:
                # If no leaf node data, just show score with unknown components
                print(f"{move_str:25} | {minimax_score:6} | {'?':6} | {'?':6} | {'?':6} | {'?':5} | {is_best:^6} | {'?'}")
    
    # Print the board for visualization
    print("\nBoard state:")
    for i in range(5):
        row = ""
        for j in range(5):
            piece = game_state_1.board[i][j]
            if piece is None:
                row += "_ "
            elif piece["type"] == "TIGER":
                row += "T "
            else:
                row += "G "
        print(row.strip())
    
    print(f"\nCurrent turn: {game_state_1.turn}")
    print(f"Phase: {game_state_1.phase}")
    print(f"Goats placed: {game_state_1.goats_placed}")

if __name__ == "__main__":
    test_minimax_agent()
    
# Add a pytest-compatible test function
def test_minimax_function():
    """Pytest-compatible test function for the minimax agent."""
    # Test with a smaller depth for faster execution during automated testing
    test_minimax_agent(depth=3)
    # No assertions needed as this is primarily a visual test
    # The test passes if no exceptions are raised