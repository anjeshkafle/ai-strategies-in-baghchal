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
    """
    Evaluates the fitness of chromosomes in the genetic algorithm.
    Uses tournament-based approach and multiprocessing.
    """
    
    def __init__(self, games_per_evaluation: int = 4, search_depth: int = 3, num_processes: int = None):
        """
        Initialize the fitness evaluator.
        
        Args:
            games_per_evaluation: Number of games to play for each evaluation
            search_depth: Search depth for minimax agent
            num_processes: Number of parallel processes to use (default: CPU count - 1)
        """
        self.games_per_evaluation = games_per_evaluation
        self.search_depth = search_depth
        
        # Set up multiprocessing
        if num_processes is None:
            import multiprocessing
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_processes = num_processes
        
        self.logger = logging.getLogger('fitness_evaluator')
    
    def evaluate_chromosome(self, genes: Dict[str, Any]) -> float:
        """
        Evaluate a single chromosome by playing games against the baseline agent.
        
        Args:
            genes: Parameters to evaluate
            
        Returns:
            Fitness score
        """
        # Create a partial function for the worker with the search depth
        worker_fn = partial(play_game_worker, 
                           agent_params=genes, 
                           agent_side="TIGER", 
                           seed=None, 
                           search_depth=self.search_depth)
        
        # Play as tiger
        tiger_results = [worker_fn() for _ in range(self.games_per_evaluation // 2)]
        
        # Play as goat
        worker_fn = partial(play_game_worker, 
                           agent_params=genes, 
                           agent_side="GOAT", 
                           seed=None, 
                           search_depth=self.search_depth)
        goat_results = [worker_fn() for _ in range(self.games_per_evaluation // 2)]
        
        # Calculate win rates
        tiger_win_count = sum(1 for res in tiger_results if res[0] == "TIGER")
        goat_win_count = sum(1 for res in goat_results if res[0] == "GOAT")
        
        tiger_win_rate = tiger_win_count / (self.games_per_evaluation // 2) if self.games_per_evaluation > 0 else 0
        goat_win_rate = goat_win_count / (self.games_per_evaluation // 2) if self.games_per_evaluation > 0 else 0
        
        # The fitness is the average of tiger and goat win rates
        fitness = (tiger_win_rate + goat_win_rate) / 2
        
        return fitness
    
    def evaluate_population(self, population: List[Chromosome]) -> List[float]:
        """
        Evaluate fitness for an entire population.
        
        Args:
            population: List of chromosomes to evaluate
            
        Returns:
            List of fitness scores in the same order as the population
        """
        return [self.evaluate_chromosome(chromosome.genes) for chromosome in population]
    
    def get_performance_metrics(self, chromosome: Chromosome) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a chromosome.
        
        Args:
            chromosome: Chromosome to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        # First ensure the chromosome has been evaluated
        self.evaluate_chromosome(chromosome.genes)
        
        # Get chromosome hash
        chromosome_hash = hash(json.dumps(chromosome.genes, sort_keys=True))
        
        # Return cached metrics or empty dict if not found
        return self.performance_cache.get(chromosome_hash, {}) 