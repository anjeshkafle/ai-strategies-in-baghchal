"""
Fitness evaluator for genetic algorithm optimization.
Evaluates chromosomes by simulating games with various agent configurations.
"""
import os
import sys
import time
import json
import logging
import traceback
import multiprocessing
from typing import Dict, Any, List, Tuple, Set

# Get main logger
logger = logging.getLogger('fitness_evaluator')

# Import local modules
from models.game_state import GameState
from models.minimax_agent import MinimaxAgent
from .params_manager import apply_tuned_parameters

# Define player constants to match GameState
PLAYER_TIGER = "TIGER"
PLAYER_GOAT = "GOAT"


def play_game_worker(args) -> Tuple[str, int, str]:
    """
    Worker function to play a game between agents.
    Used with multiprocessing to parallelize game simulations.
    
    Args:
        args: Tuple containing (genes, play_as, seed, depth)
        
    Returns:
        Tuple of (winner, move_count, side_played_as)
    """
    try:
        # Extract arguments
        genes, play_as, seed, depth = args
        
        # Create game state
        game_state = GameState()
        
        # Create agents
        if play_as == PLAYER_TIGER:
            # Tuned agent plays as tiger
            tiger_agent = MinimaxAgent(max_depth=depth)
            apply_tuned_parameters(tiger_agent, genes)
            goat_agent = MinimaxAgent(max_depth=depth)  # Default agent as goat
        else:
            # Tuned agent plays as goat
            tiger_agent = MinimaxAgent(max_depth=depth)  # Default agent as tiger
            goat_agent = MinimaxAgent(max_depth=depth)
            apply_tuned_parameters(goat_agent, genes)
        
        # Play the game
        max_moves = 200  # Avoid infinite games
        
        # Time tracking
        move_count = 0
        start_time = time.time()
        
        # Main game loop
        while not game_state.is_terminal() and move_count < max_moves:
            # Track move start time
            move_start = time.time()
            
            # Get current player
            current_player = game_state.turn  # Use turn attribute directly
            
            # Select agent based on current player
            agent = tiger_agent if current_player == PLAYER_TIGER else goat_agent
            
            # Get agent's move
            move = agent.get_move(game_state)
            
            # Apply move
            game_state.apply_move(move)
            move_count += 1
        
        # Determine winner
        if game_state.is_terminal():
            winner = game_state.get_winner()
        elif move_count >= max_moves:
            # Check if tigers are close to winning
            if game_state.goats_captured >= 3:
                winner = PLAYER_TIGER
            # Check if goats are in a strong position
            elif game_state.goats_placed == game_state.TOTAL_GOATS and game_state.goats_captured <= 1:
                winner = PLAYER_GOAT
            else:
                winner = "DRAW"
        else:
            # Game ended due to threefold repetition
            winner = "DRAW"
        
        return (winner, move_count, play_as)
        
    except Exception as e:
        logger.error(f"Error in play_game_worker: {e}")
        return ("ERROR", 0, play_as)


class FitnessEvaluator:
    """
    Evaluates fitness of chromosomes by simulating games.
    Uses multiprocessing for parallel evaluation.
    """
    
    def __init__(self, games_per_evaluation: int = 10, search_depth: int = 3, num_processes: int = None):
        """
        Initialize the fitness evaluator.
        
        Args:
            games_per_evaluation: Number of games to play for each evaluation
            search_depth: Search depth for MinimaxAgent
            num_processes: Number of processes to use for parallel evaluation
        """
        self.games_per_evaluation = games_per_evaluation
        self.search_depth = search_depth
        
        # Set up parallel processing
        if num_processes is None or num_processes <= 0:
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_processes = num_processes
        
        # Cache for performance metrics
        self.performance_cache = {}
        
        # Track last win rates
        self.last_tiger_win_rate = 0.0
        self.last_goat_win_rate = 0.0
        
        logger.info(f"FitnessEvaluator initialized with {games_per_evaluation} games, depth {search_depth}")
    
    def evaluate_chromosome(self, genes: Dict[str, Any]) -> float:
        """
        Evaluate a single chromosome by playing games against the baseline agent.
        
        Args:
            genes: Parameters to evaluate
            
        Returns:
            Fitness score
        """
        eval_start = time.time()
        
        # Try to use multiprocessing pool first
        try:
            # Create lists of games to play
            tiger_games = [(genes, "TIGER", i, self.search_depth) 
                          for i in range(self.games_per_evaluation // 2)]
            goat_games = [(genes, "GOAT", i + 100, self.search_depth)  # Use different seeds
                         for i in range(self.games_per_evaluation // 2)]
            
            all_games = tiger_games + goat_games
            
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.map(play_game_worker, all_games)
            
            # Separate results by side
            tiger_results = results[:len(tiger_games)]
            goat_results = results[len(tiger_games):]
            
        except Exception as e:
            logger.error(f"Multiprocessing evaluation failed: {e}")
            
            # Fall back to sequential evaluation
            logger.warning("Falling back to sequential evaluation")
            
            # Play as tiger
            tiger_results = []
            for i in range(self.games_per_evaluation // 2):
                result = play_game_worker((genes, "TIGER", i, self.search_depth))
                tiger_results.append(result)
            
            # Play as goat
            goat_results = []
            for i in range(self.games_per_evaluation // 2):
                result = play_game_worker((genes, "GOAT", i + 100, self.search_depth))
                goat_results.append(result)
        
        # Calculate win rates
        tiger_win_count = sum(1 for res in tiger_results if res[0] == "TIGER")
        goat_win_count = sum(1 for res in goat_results if res[0] == "GOAT")
        
        tiger_win_rate = tiger_win_count / (self.games_per_evaluation // 2) if self.games_per_evaluation > 0 else 0
        goat_win_rate = goat_win_count / (self.games_per_evaluation // 2) if self.games_per_evaluation > 0 else 0
        
        # Store win rates
        self.last_tiger_win_rate = tiger_win_rate
        self.last_goat_win_rate = goat_win_rate
        
        # The fitness is the average of tiger and goat win rates
        fitness = (tiger_win_rate + goat_win_rate) / 2
        
        # Store performance metrics
        chromosome_hash = hash(json.dumps(genes, sort_keys=True))
        self.performance_cache[chromosome_hash] = {
            'tiger_win_rate': tiger_win_rate,
            'goat_win_rate': goat_win_rate,
            'total_games': self.games_per_evaluation,
            'evaluation_time': time.time() - eval_start
        }
        
        logger.info(f"Chromosome evaluated with fitness {fitness:.2f} (Tiger: {tiger_win_rate:.2f}, Goat: {goat_win_rate:.2f})")
        
        return fitness
    
    def evaluate_population(self, population: List) -> List[float]:
        """
        Evaluate fitness for an entire population.
        
        Args:
            population: List of chromosomes to evaluate
            
        Returns:
            List of fitness scores in the same order as the population
        """
        logger.info(f"Evaluating population with {len(population)} chromosomes")
        pop_start = time.time()
        
        results = [self.evaluate_chromosome(chromosome.genes) for chromosome in population]
        
        logger.info(f"Population evaluation completed in {time.time() - pop_start:.2f}s")
        return results
    
    def get_performance_metrics(self, chromosome) -> Dict[str, Any]:
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