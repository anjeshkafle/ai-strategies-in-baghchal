#!/usr/bin/env python3
import sys
import os
import unittest
from pprint import pprint

# Add parent directory to path to make imports work in test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimax_agent import MinimaxAgent
from models.game_state import GameState

class TestMinimaxEvaluation(unittest.TestCase):
    """Test the MinimaxAgent's evaluation function with different parameter sets."""

    def setUp(self):
        """Set up test cases with specific board configurations that highlight parameter differences."""
        # Board with many closed spaces for tigers
        self.closed_spaces_board = [
            "GGGGG",
            "G_G_G",
            "GGTGG",
            "G_G_G",
            "GGGGG"
        ]
        # Board with tigers in various positions (good and bad)
        self.tiger_position_board = [
            "T___T",
            "_____",
            "__T__",
            "_____",
            "T___T"
        ]
        # Board with goats on edges and center
        self.goat_edge_board = [
            "G_G_G",
            "_____",
            "G_G_G",
            "_____",
            "G_G_G"
        ]

    def _string_board_to_game_state(self, string_board, phase="MOVEMENT", turn="TIGER", 
                                    goats_placed=0, goats_captured=0):
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
    
    def _print_board(self, board):
        """Print the board in a human-readable format."""
        print("\nBoard:")
        for row in board:
            print("".join([cell if cell in ["T", "G"] else "_" for cell in row]))

    def test_closed_spaces_evaluation(self):
        """Test that the closed spaces evaluation differs with tuned parameters."""
        # Create the game state from the closed spaces board
        state = self._string_board_to_game_state(self.closed_spaces_board, goats_placed=20)
        self._print_board(self.closed_spaces_board)

        # Create agents with default and tuned parameters
        default_agent = MinimaxAgent(max_depth=3, useTunedParams=False)
        tuned_agent = MinimaxAgent(max_depth=3, useTunedParams=True)

        # Evaluate with both agents
        default_score = default_agent.evaluate(state)
        tuned_score = tuned_agent.evaluate(state)

        # Get the closed spaces weight values
        default_weight = default_agent.closed_spaces_weight
        if tuned_agent._using_tuned_params and 'closed_space_weight_factor' in tuned_agent.tuned_factors:
            tuned_factor = tuned_agent.tuned_factors['closed_space_weight_factor']
        else:
            tuned_factor = 1.0

        # Print results
        print("\nClosed Spaces Board Evaluation:")
        print(f"Default closed spaces weight: {default_weight}")
        print(f"Tuned closed spaces factor: {tuned_factor}")
        print(f"Default evaluation score: {default_score}")
        print(f"Tuned evaluation score: {tuned_score}")
        print(f"Difference: {abs(default_score - tuned_score)}")

        # The scores should differ significantly for a board with many closed spaces
        self.assertNotEqual(default_score, tuned_score, 
                           "Scores should differ for closed spaces board when using different parameters")

    def test_tiger_position_evaluation(self):
        """Test that the tiger position evaluation differs with tuned parameters."""
        # Create the game state from the tiger position board
        state = self._string_board_to_game_state(self.tiger_position_board, goats_placed=0)
        self._print_board(self.tiger_position_board)

        # Create agents with default and tuned parameters
        default_agent = MinimaxAgent(max_depth=3, useTunedParams=False)
        tuned_agent = MinimaxAgent(max_depth=3, useTunedParams=True)

        # Evaluate with both agents
        default_score = default_agent.evaluate(state)
        tuned_score = tuned_agent.evaluate(state)

        # Print the position weight factors
        default_weight = default_agent.dispersion_weight
        if tuned_agent._using_tuned_params and 'position_weight_factor' in tuned_agent.tuned_factors:
            tuned_factor = tuned_agent.tuned_factors['position_weight_factor']
        else:
            tuned_factor = "Not using tuned factors"

        # Print results
        print("\nTiger Position Board Evaluation:")
        print(f"Default dispersion weight: {default_weight}")
        print(f"Tuned position weight factor: {tuned_factor}")
        print(f"Default evaluation score: {default_score}")
        print(f"Tuned evaluation score: {tuned_score}")
        print(f"Difference: {abs(default_score - tuned_score)}")

        # The scores should differ for a board emphasizing tiger positions
        self.assertNotEqual(default_score, tuned_score, 
                           "Scores should differ for tiger position board when using different parameters")

    def test_goat_edge_evaluation(self):
        """Test that the goat edge preference evaluation differs with tuned parameters."""
        # Create the game state with different goats placed values to test dynamic equilibrium
        early_state = self._string_board_to_game_state(self.goat_edge_board, goats_placed=5)
        mid_state = self._string_board_to_game_state(self.goat_edge_board, goats_placed=10)
        late_state = self._string_board_to_game_state(self.goat_edge_board, goats_placed=18)
        
        # Display the board
        self._print_board(self.goat_edge_board)

        # Create agents with default and tuned parameters
        default_agent = MinimaxAgent(max_depth=3, useTunedParams=False)
        tuned_agent = MinimaxAgent(max_depth=3, useTunedParams=True)

        # Evaluate for early, mid, and late game with both agents
        default_early = default_agent.evaluate(early_state)
        tuned_early = tuned_agent.evaluate(early_state)
        
        default_mid = default_agent.evaluate(mid_state)
        tuned_mid = tuned_agent.evaluate(mid_state)
        
        default_late = default_agent.evaluate(late_state)
        tuned_late = tuned_agent.evaluate(late_state)

        # Print the edge weight factors
        default_weight = default_agent.edge_weight
        if tuned_agent._using_tuned_params and 'edge_weight_factor' in tuned_agent.tuned_factors:
            tuned_factor = tuned_agent.tuned_factors['edge_weight_factor']
        else:
            tuned_factor = "Not using tuned factors"

        # Print results
        print("\nGoat Edge Preference Evaluation:")
        print(f"Default edge weight: {default_weight}")
        print(f"Tuned edge weight factor: {tuned_factor}")
        print("\nEarly game (5 goats placed):")
        print(f"Default evaluation: {default_early}")
        print(f"Tuned evaluation: {tuned_early}")
        print(f"Difference: {abs(default_early - tuned_early)}")
        
        print("\nMid game (10 goats placed):")
        print(f"Default evaluation: {default_mid}")
        print(f"Tuned evaluation: {tuned_mid}")
        print(f"Difference: {abs(default_mid - tuned_mid)}")
        
        print("\nLate game (18 goats placed):")
        print(f"Default evaluation: {default_late}")
        print(f"Tuned evaluation: {tuned_late}")
        print(f"Difference: {abs(default_late - tuned_late)}")

        # The scores should differ, and the difference should vary by game stage
        self.assertNotEqual(default_early, tuned_early, 
                           "Early game scores should differ when using different parameters")
        self.assertNotEqual(default_mid, tuned_mid, 
                           "Mid game scores should differ when using different parameters")
        self.assertNotEqual(default_late, tuned_late, 
                           "Late game scores should differ when using different parameters")

if __name__ == "__main__":
    unittest.main() 