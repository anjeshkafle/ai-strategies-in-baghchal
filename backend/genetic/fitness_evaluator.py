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
import random

# Get main logger
logger = logging.getLogger('fitness_evaluator')

# Import local modules
from models.game_state import GameState
from models.minimax_agent import MinimaxAgent
from .params_manager import apply_tuned_parameters

# Define player constants to match GameState
PLAYER_TIGER = "TIGER"
PLAYER_GOAT = "GOAT"

# Console symbols for gamified display
SYMBOLS = {
    "tiger": "🐯",
    "goat": "🐐",
    "vs": "⚔️",
    "win": "✅",
    "loss": "❌",
    "draw": "🔄"
}


def get_state_hash(state: GameState) -> str:
    """
    Create a hash representation of the game state for repetition detection.
    
    Args:
        state: Current game state
        
    Returns:
        String hash of the board state
    """
    # Only track repetition during movement phase
    if state.phase == "MOVEMENT":
        # Convert board to a string representation
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
        return f"{board_str}_{state.turn}"
    else:
        # During placement phase, include goats_placed to ensure uniqueness
        return f"PLACEMENT_{state.goats_placed}_{state.turn}"


def play_game_worker(args) -> Tuple[str, int, str, Dict]:
    """
    Worker function to play a game between agents.
    Used with multiprocessing to parallelize game simulations.
    
    Args:
        args: Tuple containing (genes, play_as, seed, depth, game_id)
        
    Returns:
        Tuple of (winner, move_count, side_played_as, game_info)
    """
    try:
        # Extract arguments
        genes, play_as, seed, depth, game_id = args
        
        # Create game state
        game_state = GameState()
        
        # Create agents
        if play_as == PLAYER_TIGER:
            # Tuned agent plays as tiger
            tiger_agent = MinimaxAgent(max_depth=depth)
            apply_tuned_parameters(tiger_agent, genes)
            goat_agent = MinimaxAgent(max_depth=depth)  # Default agent as goat
            matchup = f"{SYMBOLS['tiger']} Tuned vs Default {SYMBOLS['goat']}"
        else:
            # Tuned agent plays as goat
            tiger_agent = MinimaxAgent(max_depth=depth)  # Default agent as tiger
            goat_agent = MinimaxAgent(max_depth=depth)
            apply_tuned_parameters(goat_agent, genes)
            matchup = f"{SYMBOLS['tiger']} Default vs Tuned {SYMBOLS['goat']}"
        
        # Print matchup info
        print(f"Game {game_id} starting: {matchup}")
        
        # Play the game
        max_moves = 200  # Avoid infinite games
        
        # Time tracking
        move_count = 0
        start_time = time.time()
        
        # Track visited states for threefold repetition
        visited_states = {}  # Format: {state_hash: count}
        
        # Main game loop
        while not game_state.is_terminal() and move_count < max_moves:
            # Check for threefold repetition in movement phase
            if game_state.phase == "MOVEMENT":
                state_hash = get_state_hash(game_state)
                visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                if visited_states[state_hash] >= 3:
                    # Game ends in a draw due to threefold repetition
                    print(f"Threefold repetition detected in game {game_id} after {move_count} moves")
                    winner = "DRAW"
                    reason = "THREEFOLD_REPETITION"
                    break
            
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
        game_time = time.time() - start_time
        
        if game_state.is_terminal():
            winner = game_state.get_winner()
            reason = "STANDARD"
        elif 'winner' in locals():  # Threefold repetition already set the winner
            pass  # Keep the previously set winner and reason
        elif move_count >= max_moves:
            # Check if tigers are close to winning
            if game_state.goats_captured >= 3:
                winner = PLAYER_TIGER
                reason = "MOVE_LIMIT_TIGERS_AHEAD"
            # Check if goats are in a strong position
            elif game_state.goats_placed == game_state.TOTAL_GOATS and game_state.goats_captured <= 1:
                winner = PLAYER_GOAT
                reason = "MOVE_LIMIT_GOATS_AHEAD"
            else:
                winner = "DRAW"
                reason = "MOVE_LIMIT"
        
        # Create result symbol
        if winner == "DRAW":
            result_symbol = SYMBOLS["draw"]
        elif winner == play_as:
            result_symbol = SYMBOLS["win"]
        else:
            result_symbol = SYMBOLS["loss"]
        
        # Game info for display
        game_info = {
            "matchup": matchup,
            "winner": winner,
            "reason": reason if 'reason' in locals() else "UNKNOWN",
            "moves": move_count,
            "time": game_time,
            "goats_captured": game_state.goats_captured,
            "result_symbol": result_symbol
        }
        
        # Print completion info
        print(f"Game {game_id} complete: {matchup} {result_symbol} Winner: {winner} ({move_count} moves, {game_time:.1f}s, {game_info['reason']})")
        
        return (winner, move_count, play_as, game_info)
        
    except Exception as e:
        logger.error(f"Error in play_game_worker: {e}")
        return ("ERROR", 0, play_as, {"error": str(e)})


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
        
        # Game ID counter
        self.game_counter = 0
        
        logger.info(f"FitnessEvaluator initialized with {games_per_evaluation} games, depth {search_depth}")
        print(f"\n{SYMBOLS['tiger']} {SYMBOLS['vs']} {SYMBOLS['goat']} Genetic Tuning: {games_per_evaluation} games per chromosome, depth {search_depth}")
    
    def evaluate_chromosome(self, genes: Dict[str, Any]) -> float:
        """
        Evaluate a single chromosome by playing games against the baseline agent.
        
        Args:
            genes: Parameters to evaluate
            
        Returns:
            Fitness score
        """
        eval_start = time.time()
        print(f"\n--- Evaluating Chromosome #{hash(json.dumps(genes, sort_keys=True)) % 1000:03d} ---")
        
        # Create lists of games to play
        tiger_games = [(genes, "TIGER", i, self.search_depth, f"{self.game_counter + i + 1}") 
                      for i in range(self.games_per_evaluation // 2)]
        goat_games = [(genes, "GOAT", i + 100, self.search_depth, f"{self.game_counter + i + (self.games_per_evaluation // 2) + 1}")
                     for i in range(self.games_per_evaluation // 2)]
        
        # Update game counter
        self.game_counter += self.games_per_evaluation
        
        all_games = tiger_games + goat_games
        
        # All games are processed in a single batch, but in parallel
        print(f"Running {len(all_games)} games in parallel batch with {self.num_processes} processes")
        
        # Try to use multiprocessing pool first
        try:
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.map(play_game_worker, all_games)
            
            # Separate results by side
            tiger_results = results[:len(tiger_games)]
            goat_results = results[len(tiger_games):]
            
        except Exception as e:
            logger.error(f"Multiprocessing evaluation failed: {e}")
            
            # Fall back to sequential evaluation
            logger.warning("Falling back to sequential evaluation")
            print("⚠️ Parallel processing failed. Running games sequentially.")
            
            # Play as tiger (sequentially)
            tiger_results = []
            for game_args in tiger_games:
                result = play_game_worker(game_args)
                tiger_results.append(result)
            
            # Play as goat (sequentially)
            goat_results = []
            for game_args in goat_games:
                result = play_game_worker(game_args)
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
        
        # Analyze game results for debugging and transparency
        repetition_count = sum(1 for res in tiger_results + goat_results if res[3].get("reason") == "THREEFOLD_REPETITION")
        move_limit_count = sum(1 for res in tiger_results + goat_results if "MOVE_LIMIT" in res[3].get("reason", ""))
        
        # Store performance metrics
        chromosome_hash = hash(json.dumps(genes, sort_keys=True))
        self.performance_cache[chromosome_hash] = {
            'tiger_win_rate': tiger_win_rate,
            'goat_win_rate': goat_win_rate,
            'total_games': self.games_per_evaluation,
            'evaluation_time': time.time() - eval_start,
            'repetition_draws': repetition_count,
            'move_limit_games': move_limit_count
        }
        
        # Pretty console output summary
        tiger_stats = f"{SYMBOLS['tiger']} Win rate: {tiger_win_rate:.2f} ({tiger_win_count}/{self.games_per_evaluation//2})"
        goat_stats = f"{SYMBOLS['goat']} Win rate: {goat_win_rate:.2f} ({goat_win_count}/{self.games_per_evaluation//2})"
        time_stats = f"⏱️ {time.time() - eval_start:.1f}s"
        
        # Include repetition info
        if repetition_count > 0:
            rep_info = f"| 🔄 Repetitions: {repetition_count}"
        else:
            rep_info = ""
            
        print(f"\n{tiger_stats} | {goat_stats} | Fitness: {fitness:.2f} | {time_stats} {rep_info}")
        print("-" * 60)
        
        logger.info(f"Chromosome evaluated with fitness {fitness:.2f} (Tiger: {tiger_win_rate:.2f}, Goat: {goat_win_rate:.2f}, Repetitions: {repetition_count})")
        
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
        print(f"\n{'='*20} Population Evaluation {'='*20}")
        print(f"Evaluating {len(population)} chromosomes")
        pop_start = time.time()
        
        # Evaluate one chromosome at a time
        # Each chromosome runs all its games in parallel, but we don't evaluate
        # multiple chromosomes simultaneously
        results = []
        for i, chromosome in enumerate(population):
            print(f"\nChromosome {i+1}/{len(population)}")
            fitness = self.evaluate_chromosome(chromosome.genes)
            results.append(fitness)
        
        elapsed = time.time() - pop_start
        logger.info(f"Population evaluation completed in {elapsed:.2f}s")
        print(f"\nPopulation evaluation complete in {elapsed:.2f}s")
        print('='*60)
        
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