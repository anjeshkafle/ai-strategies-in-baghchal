"""
Core genetic algorithm implementation for tuning MinimaxAgent parameters.
"""
import os
import json
import time
import random
import logging
import math
import statistics
import numpy as np
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from .chromosome import Chromosome
from .fitness_evaluator import FitnessEvaluator
from .params_manager import save_tuned_parameters
from .utils import log_generation_to_csv, calculate_population_diversity
import pickle


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for MinimaxAgent parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the genetic optimizer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('genetic_optimizer')
        
        # Population parameters
        self.population_size = config.get("population_size", 10)
        self.elitism_count = config.get("elitism_count", 2)
        self.tournament_size = config.get("tournament_size", 3)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.mutation_magnitude = config.get("mutation_magnitude", 0.2)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.max_generations = config.get("generations", 20)
        self.save_interval = config.get("save_interval", 1)
        self.max_time = config.get("max_execution_time", float('inf'))
        
        # ALWAYS use backend/tuned_params regardless of config
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tuned_params")
        
        # File to store population for resuming
        self.population_file = os.path.join(self.output_dir, "population.pkl")
        self.generation_file = os.path.join(self.output_dir, "generation_count.txt")
        
        # Ensure output directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating output directory {self.output_dir}: {e}")
            raise
        
        # Search depth for agents
        self.search_depth = config.get("search_depth", 3)
        
        # Evaluation parameters
        self.games_per_evaluation = config.get("games_per_evaluation", 4)
        
        # Set up parallel processing
        self.num_processes = config.get("parallel_processes", 0)
        if self.num_processes <= 0:
            self.num_processes = max(1, multiprocessing.cpu_count() - 1)
        
        # Initialize the fitness evaluator
        self.evaluator = FitnessEvaluator(
            games_per_evaluation=self.games_per_evaluation,
            search_depth=self.search_depth,
            num_processes=self.num_processes
        )
        
        # Store fitness history
        self.fitness_history = []
        self.best_chromosome = None
        self.start_time = None
        self.starting_generation = 0  # Initialize to 0, will be updated if resuming
    
    def _load_previous_run(self) -> Tuple[Optional[List[Chromosome]], int]:
        """
        Load population and generation count from previous run.
        
        Returns:
            Tuple containing population (or None) and generation count
        """
        population = None
        generation = 0
        
        # Try to load generation count
        if os.path.exists(self.generation_file):
            try:
                with open(self.generation_file, 'r') as f:
                    generation = int(f.read().strip())
                self.logger.info(f"Loaded generation count: {generation}")
            except Exception as e:
                self.logger.warning(f"Failed to load generation count: {e}")
        
        # Try to load population
        if os.path.exists(self.population_file):
            try:
                with open(self.population_file, 'rb') as f:
                    population = pickle.load(f)
                self.logger.info(f"Loaded population from previous run with {len(population)} chromosomes")
            except Exception as e:
                self.logger.warning(f"Failed to load population: {e}")
                population = None
        
        return population, generation
    
    def _save_population(self, population: List[Chromosome], generation: int):
        """
        Save population and generation count for potential resumption.
        
        Args:
            population: List of chromosomes to save
            generation: Current generation number
        """
        try:
            with open(self.population_file, 'wb') as f:
                pickle.dump(population, f)
            
            with open(self.generation_file, 'w') as f:
                f.write(str(generation))
                
            self.logger.info(f"Saved population and generation count for potential resumption")
        except Exception as e:
            self.logger.warning(f"Failed to save population: {e}")
    
    def run(self) -> Chromosome:
        """Run the genetic algorithm until completion."""
        # Check for previous run to resume
        loaded_population, loaded_generation = self._load_previous_run()
        
        # Initialize or resume population
        if loaded_population and loaded_generation > 0:
            population = loaded_population
            self.starting_generation = loaded_generation
            target_generation = loaded_generation + self.max_generations
            self.logger.info(f"Resuming optimization from generation {self.starting_generation}")
            self.logger.info(f"Will run for {self.max_generations} more generations (up to generation {target_generation})")
        else:
            # Initialize fresh population
            population = self._initialize_population()
            self.starting_generation = 0
            target_generation = self.max_generations
            self.logger.info("Starting new optimization run with fresh population")
        
        self.start_time = time.time()
        self.logger.info(f"Starting genetic optimization with {self.population_size} chromosomes "
                         f"for up to {self.max_time} seconds")
        
        # Clear fitness history for new run
        self.fitness_history = []
        
        # Run for specified number of generations
        for generation in range(self.starting_generation, target_generation):
            gen_start_time = time.time()
            
            # Evaluate population
            for chromosome in population:
                if chromosome.fitness is None:  # Only evaluate if not already evaluated
                    chromosome.fitness = self.evaluator.evaluate_chromosome(chromosome.genes)
            
            # Set a default value for chromosomes with None fitness
            for chromosome in population:
                if chromosome.fitness is None:
                    chromosome.fitness = 0.0
            
            # Sort by fitness (higher is better)
            self._safe_sort_population(population)
            
            # Update best chromosome
            if self.best_chromosome is None or population[0].fitness > self.best_chromosome.fitness:
                self.best_chromosome = population[0].clone()
                self.logger.info(f"New best chromosome found with fitness {self.best_chromosome.fitness:.4f}")
            
            # Calculate statistics
            fitnesses = [c.fitness for c in population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            diversity = self._calculate_diversity(population)
            
            # Store history
            gen_data = {
                'generation': generation + 1,
                'best_fitness': population[0].fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'diversity': diversity,
                'timestamp': time.time(),
                'elapsed_time': time.time() - gen_start_time,
                'total_time': time.time() - self.start_time
            }
            
            # Add tiger and goat win rates from the evaluator
            if hasattr(self.evaluator, "last_tiger_win_rate") and hasattr(self.evaluator, "last_goat_win_rate"):
                gen_data['best_tiger_win_rate'] = self.evaluator.last_tiger_win_rate
                gen_data['best_goat_win_rate'] = self.evaluator.last_goat_win_rate
            else:
                # Try to get win rates from the best chromosome
                tiger_rate, goat_rate = self._get_win_rates(population[0])
                gen_data['best_tiger_win_rate'] = tiger_rate
                gen_data['best_goat_win_rate'] = goat_rate
            
            # Save generation data to CSV for plotting
            log_generation_to_csv(self.output_dir, gen_data)
            
            # Also save detailed generation data
            self._save_generation_results(population, generation + 1)
            
            # Append to history list
            self.fitness_history.append(gen_data)
            
            # Save fitness history after each generation
            history_path = os.path.join(self.output_dir, "fitness_history.json")
            try:
                with open(history_path, 'w') as f:
                    json.dump(self.fitness_history, f, indent=2)
                self.logger.debug(f"Saved fitness history to {history_path} with {len(self.fitness_history)} generations")
            except Exception as e:
                self.logger.error(f"Error saving fitness history: {e}")
            
            # Log progress
            self.logger.info(f"Generation {generation + 1}/{target_generation}: "
                             f"Best={population[0].fitness:.4f}, Avg={avg_fitness:.4f}, "
                             f"Min={min_fitness:.4f}, Diversity={diversity:.4f}, "
                             f"Time={time.time() - gen_start_time:.2f}s, "
                             f"Total={time.time() - self.start_time:.2f}s")
            
            # Save best chromosome at intervals
            if (generation + 1) % self.save_interval == 0 or generation + 1 == target_generation:
                self._save_best_chromosome(population[0], generation + 1)
            
            # Save current population for potential resumption
            self._save_population(population, generation + 1)
            
            # Create next generation if not the last one
            if generation + 1 < target_generation:
                population = self._create_next_generation(population)
        
        # Save final best chromosome
        self._save_best_chromosome(population[0])
        
        # Ensure fitness history is saved again at the end
        if self.fitness_history:
            history_path = os.path.join(self.output_dir, "fitness_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.fitness_history, f, indent=2)
            self.logger.info(f"Saved final fitness history with {len(self.fitness_history)} generations")
        
        # Return best chromosome
        return population[0]
    
    def run_with_time_limit(self, max_time_seconds: float) -> Chromosome:
        """Run the algorithm with a time limit."""
        # Check for previous run to resume
        loaded_population, loaded_generation = self._load_previous_run()
        
        # Initialize or resume population
        if loaded_population and loaded_generation > 0:
            population = loaded_population
            self.starting_generation = loaded_generation
            target_generation = loaded_generation + self.max_generations
            self.logger.info(f"Resuming optimization from generation {self.starting_generation}")
            self.logger.info(f"Will run for up to {self.max_generations} more generations (up to generation {target_generation})")
        else:
            # Initialize fresh population
            population = self._initialize_population()
            self.starting_generation = 0
            target_generation = self.max_generations
            self.logger.info("Starting new optimization run with fresh population")
        
        self.start_time = time.time()
        self.logger.info(f"Starting genetic optimization with {self.population_size} chromosomes "
                         f"for up to {max_time_seconds} seconds")
        
        # Clear fitness history for new run
        self.fitness_history = []
        
        # Run generations until time limit is reached
        generation = self.starting_generation
        while time.time() - self.start_time < max_time_seconds and generation < target_generation:
            gen_start_time = time.time()
            
            # Evaluate population
            for chromosome in population:
                if chromosome.fitness is None:  # Only evaluate if not already evaluated
                    chromosome.fitness = self.evaluator.evaluate_chromosome(chromosome.genes)
            
            # Set a default value for chromosomes with None fitness
            for chromosome in population:
                if chromosome.fitness is None:
                    chromosome.fitness = 0.0
            
            # Sort by fitness (higher is better)
            self._safe_sort_population(population)
            
            # Update best chromosome
            if self.best_chromosome is None or population[0].fitness > self.best_chromosome.fitness:
                self.best_chromosome = population[0].clone()
                self.logger.info(f"New best chromosome found with fitness {self.best_chromosome.fitness:.4f}")
            
            # Calculate statistics
            fitnesses = [c.fitness for c in population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            diversity = self._calculate_diversity(population)
            
            # Store history
            gen_data = {
                'generation': generation + 1,
                'best_fitness': population[0].fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'diversity': diversity,
                'timestamp': time.time(),
                'elapsed_time': time.time() - gen_start_time,
                'total_time': time.time() - self.start_time
            }
            
            # Add tiger and goat win rates from the evaluator
            if hasattr(self.evaluator, "last_tiger_win_rate") and hasattr(self.evaluator, "last_goat_win_rate"):
                gen_data['best_tiger_win_rate'] = self.evaluator.last_tiger_win_rate
                gen_data['best_goat_win_rate'] = self.evaluator.last_goat_win_rate
            else:
                # Try to get win rates from the best chromosome
                tiger_rate, goat_rate = self._get_win_rates(population[0])
                gen_data['best_tiger_win_rate'] = tiger_rate
                gen_data['best_goat_win_rate'] = goat_rate
            
            # Save generation data to CSV for plotting
            log_generation_to_csv(self.output_dir, gen_data)
            
            # Also save detailed generation data
            self._save_generation_results(population, generation + 1)
            
            # Append to history list
            self.fitness_history.append(gen_data)
            
            # Save fitness history after each generation
            history_path = os.path.join(self.output_dir, "fitness_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.fitness_history, f, indent=2)
            self.logger.debug(f"Saved fitness history to {history_path} with {len(self.fitness_history)} generations")
            
            # Log progress
            self.logger.info(f"Generation {generation + 1}/{target_generation}: "
                             f"Best={population[0].fitness:.4f}, Avg={avg_fitness:.4f}, "
                             f"Min={min_fitness:.4f}, Diversity={diversity:.4f}, "
                             f"Time={time.time() - gen_start_time:.2f}s, "
                             f"Total={time.time() - self.start_time:.2f}s")
            
            # Save best chromosome at intervals
            if (generation + 1) % self.save_interval == 0:
                self._save_best_chromosome(population[0], generation + 1)
            
            # Save current population for potential resumption
            self._save_population(population, generation + 1)
            
            # Create next generation
            if time.time() - self.start_time < max_time_seconds - 5:  # Leave 5 seconds buffer
                population = self._create_next_generation(population)
            else:
                self.logger.info("Time limit approaching, not creating new generation.")
                break
            
            generation += 1
        
        # Sort final population by fitness
        self._safe_sort_population(population)
        
        # Save final best chromosome
        self._save_best_chromosome(population[0])
        
        # Save entire population for potential resumption
        self._save_population(population, generation + 1)
        
        # Ensure fitness history is saved again at the end
        if self.fitness_history:
            history_path = os.path.join(self.output_dir, "fitness_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.fitness_history, f, indent=2)
            self.logger.info(f"Saved final fitness history with {len(self.fitness_history)} generations")
        
        self.logger.info(f"Optimization complete. Best fitness: {population[0].fitness:.4f}, "
                         f"Total time: {time.time() - self.start_time:.2f}s, "
                         f"Generations: {generation - self.starting_generation + 1}")
        
        return population[0]
    
    def _initialize_population(self) -> List[Chromosome]:
        """
        Initialize the population with random chromosomes.
        
        Returns:
            List of initialized chromosomes
        """
        import random
        import time
        from .chromosome import Chromosome
        
        debug_logger = logging.getLogger('ga_debug')
        debug_logger.info(f"Initializing population with {self.population_size} chromosomes")
        
        population = []
        for i in range(self.population_size):
            # Use different random seeds for each chromosome
            # This ensures diversity even with small populations
            random.seed(time.time() + i)
            
            # Create a new chromosome with random genes
            chromosome = Chromosome.create_random(self.config)
            
            # Ensure fitness is None initially
            chromosome.fitness = None
            
            # Add to population
            population.append(chromosome)
            
            # Debug log the chromosome to verify initialization
            debug_logger.debug(f"Initialized chromosome {i+1}/{self.population_size}: {chromosome.genes}")
        
        # Verify population has the correct size
        assert len(population) == self.population_size, f"Population size mismatch: {len(population)} != {self.population_size}"
        
        # Log diversity metric
        diversity = self._calculate_diversity(population)
        debug_logger.info(f"Initial population diversity: {diversity:.4f}")
        
        return population
    
    def _create_next_generation(self, current_population: List[Chromosome]) -> List[Chromosome]:
        """
        Create the next generation using selection, crossover, and mutation.
        
        Args:
            current_population: Current generation of chromosomes
            
        Returns:
            New generation of chromosomes
        """
        # Sort population by fitness (descending)
        sorted_population = sorted(current_population, key=lambda x: x.fitness, reverse=True)
        
        # Apply elitism (keep best chromosomes)
        new_population = [c.clone() for c in sorted_population[:self.elitism_count]]
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self._tournament_selection(current_population)
            parent2 = self._tournament_selection(current_population)
            
            # Apply crossover with probability crossover_rate
            if random.random() < self.crossover_rate:
                child1, child2 = Chromosome.crossover(parent1, parent2)
            else:
                # Otherwise, clone parents
                child1, child2 = parent1.clone(), parent2.clone()
            
            # Apply mutation
            child1.mutate(self.mutation_rate, self.mutation_magnitude)
            child2.mutate(self.mutation_rate, self.mutation_magnitude)
            
            # Add children to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """
        Select a chromosome using tournament selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected chromosome
        """
        # Randomly select tournament_size chromosomes
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Return the best one
        return max(tournament, key=lambda x: x.fitness)
    
    def _save_best_chromosome(self, chromosome: Chromosome, generation: int = None):
        """
        Save the best chromosome to file.
        
        Args:
            chromosome: Best chromosome
            generation: Generation number (optional)
        """
        # Always save to the main best_params.json
        best_path = os.path.join(self.output_dir, "best_params.json")
        chromosome.save_to_file(best_path)
        
        # If generation is provided, save a backup with generation number
        if generation is not None:
            backup_path = os.path.join(self.output_dir, f"best_params_{generation}.json")
            chromosome.save_to_file(backup_path)
        
        # Log
        self.logger.info(f"Best parameters saved to {best_path}")
    
    def _calculate_diversity(self, population: List[Chromosome]) -> float:
        """
        Calculate population diversity based on gene values.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Diversity score (0.0-1.0)
        """
        try:
            import numpy as np
            
            # If population is too small, return a default value
            if len(population) <= 1:
                return 0.0
                
            # Get all gene values in a matrix
            # First, get a list of all gene keys
            first_chrom = population[0]
            gene_keys = list(first_chrom.genes.keys())
            
            # Create a matrix of gene values
            gene_matrix = []
            for chromosome in population:
                gene_values = [chromosome.genes.get(key, 0) for key in gene_keys]
                gene_matrix.append(gene_values)
            
            # Convert to numpy array for efficient computation
            gene_matrix = np.array(gene_matrix)
            
            # Calculate standard deviation for each gene across population
            std_devs = np.std(gene_matrix, axis=0)
            
            # Calculate average standard deviation as diversity measure
            avg_std_dev = np.mean(std_devs)
            
            # Normalize by the maximum possible value for these genes
            # This is a simplification - using average range from parameter_ranges
            # For a more accurate normalization, you would need to use actual ranges
            # from config for each specific gene
            average_range = 0.1  # A default normalization factor
            
            # Normalize to 0-1 range
            diversity = min(1.0, avg_std_dev / average_range)
            
            return diversity
            
        except Exception as e:
            # If there's any error in calculation, use a fallback method
            self.logger.warning(f"Error calculating diversity: {e}, using fallback method")
            
            # Fallback: use a simpler method - just count unique chromosomes
            unique_chroms = set()
            for chrom in population:
                unique_chroms.add(str(chrom.genes))
            
            # Normalize by population size
            return len(unique_chroms) / len(population)
    
    def _save_generation_results(self, population: List[Chromosome], generation: int):
        """
        Save detailed results for a generation.
        
        Args:
            population: Current population
            generation: Generation number
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Calculate statistics
        avg_fitness = sum(c.fitness for c in sorted_pop) / len(sorted_pop)
        min_fitness = min(c.fitness for c in sorted_pop)
        diversity = self._calculate_diversity(population)
        
        # Get win rates for best chromosome
        best_tiger_win_rate, best_goat_win_rate = self._get_win_rates(sorted_pop[0])
        
        # Extract data for saving
        gen_data = {
            "generation": generation,
            "timestamp": time.time(),
            "best_fitness": sorted_pop[0].fitness,
            "avg_fitness": avg_fitness,
            "min_fitness": min_fitness,
            "tiger_win_rate": best_tiger_win_rate,
            "goat_win_rate": best_goat_win_rate,
            "diversity": diversity,
            "chromosomes": [c.to_dict() for c in sorted_pop[:10]],  # Save top 10
            "elapsed_time": time.time() - self.start_time
        }
        
        # Save to file
        gen_path = os.path.join(self.output_dir, f"generation_{generation}.json")
        with open(gen_path, 'w') as f:
            json.dump(gen_data, f, indent=2)
    
    def _get_win_rates(self, chromosome: Chromosome) -> Tuple[float, float]:
        """
        Get the win rates for the best chromosome against baseline.
        Uses cached data from the fitness evaluator.
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Tuple of (tiger_win_rate, goat_win_rate)
        """
        try:
            if not hasattr(chromosome, 'genes'):
                self.logger.error("Invalid chromosome - missing 'genes' attribute")
                return 0.0, 0.0
            
            # The fitness evaluator caches results by chromosome hash
            # Use the chromosome's genes to evaluate, not the chromosome itself
            fitness = self.evaluator.evaluate_chromosome(chromosome.genes)
            
            # If we can access the raw win rates from evaluator, use them
            if hasattr(self.evaluator, "last_tiger_win_rate") and hasattr(self.evaluator, "last_goat_win_rate"):
                return self.evaluator.last_tiger_win_rate, self.evaluator.last_goat_win_rate
            
            # Otherwise, use a rough estimate
            # Assuming fitness is approximately 50% tiger performance + 50% goat performance
            # This is an approximation based on how fitness_evaluator.py combines scores
            tiger_estimate = min(1.0, max(0.0, fitness * 0.5))
            goat_estimate = min(1.0, max(0.0, fitness * 0.5))
            
            return tiger_estimate, goat_estimate
        except AttributeError as e:
            self.logger.error(f"AttributeError in _get_win_rates: {e}")
            return 0.0, 0.0
        except TypeError as e:
            self.logger.error(f"TypeError in _get_win_rates: {e}")
            return 0.0, 0.0
        except Exception as e:
            self.logger.error(f"Error getting win rates: {e}")
            return 0.0, 0.0
    
    def _safe_sort_population(self, population: List[Chromosome]):
        """
        Safely sort population by fitness, handling None values.
        
        Args:
            population: Population to sort
        """
        # First ensure all chromosomes have a valid fitness value
        for chromosome in population:
            if chromosome.fitness is None:
                self.logger.warning(f"Found chromosome with None fitness, setting to 0.0")
                chromosome.fitness = 0.0
        
        # Now sort by fitness (higher is better)
        population.sort(key=lambda x: x.fitness, reverse=True) 