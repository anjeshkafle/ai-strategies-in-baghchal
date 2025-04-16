"""
Fitness evaluator for genetic algorithm tuning of MinimaxAgent.
Implements tournament-based evaluation of chromosomes with efficient multiprocessing.
"""
import os
import json
import time
import logging
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Tuple
from models.minimax_agent import MinimaxAgent
from models.game_state import GameState
from .params_manager import apply_tuned_parameters
from .chromosome import Chromosome


def play_game_worker(args):
    """
    Worker function for playing a single game in separate process.
    
    Args:
        args: Tuple containing (agent_params, agent_side, seed, search_depth)
        
    Returns:
        Tuple of (winner, moves_count, agent_side)
    """
    agent_params, agent_side, seed, search_depth = args
    
    # Create agents with specified search depth
    agent = MinimaxAgent(max_depth=search_depth, randomize_equal_moves=False)
    apply_tuned_parameters(agent, params=agent_params)
    
    baseline = MinimaxAgent(max_depth=search_depth, randomize_equal_moves=False)
    
    # Set up agents based on sides
    tiger_agent = agent if agent_side == "TIGER" else baseline
    goat_agent = agent if agent_side == "GOAT" else baseline
    
    # Initialize game state with seed for reproducibility
    state = GameState()
    if seed is not None:
        # Use seed for reproducible games if provided
        import random
        random.seed(seed)
    
    moves_count = 0
    max_moves = 200  # Prevent infinite games
    
    # Track visited states for threefold repetition
    visited_states = {}  # Format: {state_hash: count}
    
    # Play the game
    while not state.is_terminal() and moves_count < max_moves:
        # Get current agent
        current_agent = tiger_agent if state.turn == "TIGER" else goat_agent
        
        # Check for threefold repetition in movement phase
        if state.phase == "MOVEMENT":
            # Create a hash of the current board state
            board_str = ""
            for row in state.board:
                for cell in row:
                    if cell is None:
                        board_str += "_"
                    elif cell["type"] == "TIGER":
                        board_str += "T"
                    else:
                        board_str += "G"
            
            # Include turn in the hash
            state_hash = f"{board_str}_{state.turn}"
            
            # Check for repetition
            visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
            if visited_states[state_hash] >= 3:
                # Draw due to threefold repetition
                return "DRAW", moves_count, agent_side
        
        # Get and apply move
        try:
            move = current_agent.get_move(state)
            state.apply_move(move)
            moves_count += 1
        except Exception as e:
            # If something goes wrong, log and terminate
            print(f"Error in game: {e}")
            return "ERROR", moves_count, agent_side
    
    # Determine winner
    winner = state.get_winner()
    
    # If no winner after max moves, declare draw (counts as loss for agent)
    if winner is None:
        winner = "GOAT" if agent_side == "TIGER" else "TIGER"
    
    return winner, moves_count, agent_side


class FitnessEvaluator:
    """Evaluates chromosome fitness through self-play tournaments."""
    
    def __init__(self, config: Dict, baseline_agent: MinimaxAgent = None):
        """
        Initialize the fitness evaluator.
        
        Args:
            config: GA configuration dictionary
            baseline_agent: Optional baseline agent for comparison
        """
        self.config = config
        self.games_per_evaluation = config.get("games_per_evaluation", 10)
        self.search_depth = config.get("search_depth", 5)
        
        # Use cpu_count-1 cores by default, but allow configuration
        parallel_setting = config.get("parallel_processes", None)
        if parallel_setting is None or parallel_setting <= 0:
            # Default to cpu_count-1 if not provided or set to 0 or negative
            self.parallel_processes = max(1, multiprocessing.cpu_count() - 1)
        else:
            # Use the provided positive value
            self.parallel_processes = parallel_setting
        
        # Cache for fitness evaluations
        self.evaluation_cache = {}
        
        # Track the last win rates for reporting
        self.last_tiger_win_rate = 0.0
        self.last_goat_win_rate = 0.0
        
        # Cache for detailed performance metrics
        self.performance_cache = {}
    
    def evaluate_population(self, population: List[Chromosome]) -> List[float]:
        """
        Evaluate fitness for an entire population.
        
        Args:
            population: List of chromosomes to evaluate
            
        Returns:
            List of fitness scores in the same order as the population
        """
        return [self.evaluate_chromosome(chromosome) for chromosome in population]
    
    def evaluate_chromosome(self, chromosome: Chromosome) -> float:
        """
        Evaluate a single chromosome using self-play tournaments.
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Fitness score (normalized win rate)
        """
        # Check cache first using chromosome hash
        chromosome_hash = hash(json.dumps(chromosome.genes, sort_keys=True))
        if chromosome_hash in self.evaluation_cache:
            # Retrieve cached performance metrics too
            if chromosome_hash in self.performance_cache:
                metrics = self.performance_cache[chromosome_hash]
                self.last_tiger_win_rate = metrics.get('tiger_win_rate', 0.0)
                self.last_goat_win_rate = metrics.get('goat_win_rate', 0.0)
            
            return self.evaluation_cache[chromosome_hash]
        
        # Create a list of all games that need to be played
        # Each game is defined by a tuple of (params, agent_side, seed, search_depth)
        games_to_play = []
        
        # Add games playing as Tiger
        for i in range(self.games_per_evaluation):
            games_to_play.append((chromosome.genes, "TIGER", i, self.search_depth))
        
        # Add games playing as Goat
        for i in range(self.games_per_evaluation):
            games_to_play.append((chromosome.genes, "GOAT", i + self.games_per_evaluation, self.search_depth))
        
        # Play all games in parallel
        results = []
        
        # Use a process pool for parallel game execution
        # Windows requires the code to be in a if __name__ == '__main__' block
        # So we need a more careful approach with initialization
        try:
            # Create a multiprocessing pool and play games in parallel
            with multiprocessing.Pool(processes=self.parallel_processes) as pool:
                results = pool.map(play_game_worker, games_to_play)
        
        except Exception as e:
            # Fallback to sequential if multiprocessing fails
            print(f"Multiprocessing failed: {e}. Falling back to sequential.")
            results = [play_game_worker(game) for game in games_to_play]
        
        # Process the results
        tiger_results = [r for r in results if r[2] == "TIGER"]
        goat_results = [r for r in results if r[2] == "GOAT"]
        
        # Count wins and track game lengths
        tiger_wins = sum(1 for r in tiger_results if r[0] == "TIGER")
        goat_wins = sum(1 for r in goat_results if r[0] == "GOAT")
        
        tiger_game_lengths = [r[1] for r in tiger_results]
        goat_game_lengths = [r[1] for r in goat_results]
        
        # Calculate fitness components
        tiger_win_rate = tiger_wins / len(tiger_results) if tiger_results else 0
        goat_win_rate = goat_wins / len(goat_results) if goat_results else 0
        
        # Save win rates for reporting
        self.last_tiger_win_rate = tiger_win_rate
        self.last_goat_win_rate = goat_win_rate
        
        # Tiger win lengths - only consider winning games for average
        winning_tiger_lengths = [r[1] for r in tiger_results if r[0] == "TIGER"]
        winning_goat_lengths = [r[1] for r in goat_results if r[0] == "GOAT"]
        
        # Adjust for game length (prefer shorter wins and longer losses)
        avg_tiger_win_length = sum(winning_tiger_lengths) / len(winning_tiger_lengths) if winning_tiger_lengths else 0
        avg_goat_win_length = sum(winning_goat_lengths) / len(winning_goat_lengths) if winning_goat_lengths else 0
        
        # Normalize game lengths (0-1 where 1 is better)
        norm_tiger_length = min(1.0, max(0.0, 1.0 - (avg_tiger_win_length / 100))) if avg_tiger_win_length > 0 else 0
        norm_goat_length = min(1.0, max(0.0, avg_goat_win_length / 100)) if avg_goat_win_length > 0 else 0
        
        # Calculate overall fitness (50% win rate, 50% game length efficiency)
        tiger_fitness = 0.5 * tiger_win_rate + 0.1 * norm_tiger_length
        goat_fitness = 0.5 * goat_win_rate + 0.1 * norm_goat_length
        
        # Combined fitness (playing both sides equally)
        fitness = tiger_fitness + goat_fitness
        
        # Cache the result
        self.evaluation_cache[chromosome_hash] = fitness
        
        # Cache performance metrics
        self.performance_cache[chromosome_hash] = {
            'tiger_win_rate': tiger_win_rate,
            'goat_win_rate': goat_win_rate,
            'avg_tiger_win_length': avg_tiger_win_length,
            'avg_goat_win_length': avg_goat_win_length,
            'tiger_wins': tiger_wins,
            'goat_wins': goat_wins,
            'tiger_game_lengths': tiger_game_lengths,
            'goat_game_lengths': goat_game_lengths
        }
        
        return fitness
    
    def get_performance_metrics(self, chromosome: Chromosome) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a chromosome.
        
        Args:
            chromosome: Chromosome to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        # First ensure the chromosome has been evaluated
        self.evaluate_chromosome(chromosome)
        
        # Get chromosome hash
        chromosome_hash = hash(json.dumps(chromosome.genes, sort_keys=True))
        
        # Return cached metrics or empty dict if not found
        return self.performance_cache.get(chromosome_hash, {}) 