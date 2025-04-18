#!/usr/bin/env python3
import sys
import os
import unittest

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState

class TestMinimaxTunedParams(unittest.TestCase):
    """Test the MinimaxAgent's ability to use either tuned or default parameters."""

    def setUp(self):
        """Set up test cases with different board configurations."""
        # Board configurations
        self.boards = {
            # Early game board with tigers on corners and a few goats
            "early_game": [
                "T___T",
                "_G___",
                "__G__",
                "___G_",
                "T___T"
            ],
            # Mid game board with a threatened goat
            "mid_game": [
                "T_G__",
                "GG___",
                "_G_TG",
                "__G__",
                "T___T"
            ],
            # Late game board with many goats and some captures
            "late_game": [
                "GGGGG",
                "GG_GG",
                "GGGTG",
                "GTT_G",
                "GGTGG"
            ]
        }

    def _string_board_to_game_state(self, string_board, phase="MOVEMENT", turn="TIGER", 
                                    goats_placed=20, goats_captured=0):
        """Convert a string-based board to a GameState object."""
        # Create an empty board
        board = [[None for _ in range(5)] for _ in range(5)]
        
        # Fill the board based on the string representation
        for y, row in enumerate(string_board):
            for x, cell in enumerate(row):
                if cell == 'T':
                    board[y][x] = {"type": "TIGER"}
                elif cell == 'G':
                    board[y][x] = {"type": "GOAT"}
        
        # Create the game state
        state = GameState()
        state.board = board
        state.phase = phase
        state.turn = turn
        state.goats_placed = goats_placed
        state.goats_captured = goats_captured
        
        return state

    def test_evaluation_differs_with_tuned_params(self):
        """Test that evaluation scores differ when using tuned vs default parameters."""
        for board_name, board_config in self.boards.items():
            state = self._string_board_to_game_state(board_config)
            
            # Create agents with default and tuned parameters
            default_agent = MinimaxAgent(max_depth=3, useTunedParams=False)
            tuned_agent = MinimaxAgent(max_depth=3, useTunedParams=True)
            
            # Evaluate the same position with both agents
            default_score = default_agent.evaluate(state)
            tuned_score = tuned_agent.evaluate(state)
            
            # Print scores for reference
            print(f"\nBoard: {board_name}")
            print(f"Default parameters score: {default_score}")
            print(f"Tuned parameters score: {tuned_score}")
            print(f"Difference: {abs(default_score - tuned_score)}")
            
            # The scores should differ when using different parameters
            # This is a very basic test just to confirm the parameters are being applied
            self.assertNotEqual(default_score, tuned_score, 
                                f"Scores should differ for {board_name} board when using different parameters")
    
    def test_internal_state_flags(self):
        """Test that the internal state flags are set correctly."""
        # Create agents with default and tuned parameters
        default_agent = MinimaxAgent(max_depth=3, useTunedParams=False)
        tuned_agent = MinimaxAgent(max_depth=3, useTunedParams=True)
        
        # Check the _using_tuned_params flag
        self.assertFalse(default_agent._using_tuned_params, 
                        "Default agent should have _using_tuned_params=False")
        self.assertTrue(tuned_agent._using_tuned_params, 
                        "Tuned agent should have _using_tuned_params=True")
        
        # Check that tuned_factors and tuned_equilibrium dicts are populated for tuned agent
        self.assertTrue(len(tuned_agent.tuned_factors) > 0, 
                        "Tuned agent should have tuned_factors populated")
        self.assertTrue(len(tuned_agent.tuned_equilibrium) > 0, 
                        "Tuned agent should have tuned_equilibrium populated")
        
        # Default agent should have empty tuned_factors and tuned_equilibrium dicts
        self.assertEqual(len(default_agent.tuned_factors), 0, 
                         "Default agent should have empty tuned_factors")
        self.assertEqual(len(default_agent.tuned_equilibrium), 0, 
                         "Default agent should have empty tuned_equilibrium")

if __name__ == "__main__":
    unittest.main() 