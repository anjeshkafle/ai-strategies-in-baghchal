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
from typing import List, Dict, Any, Tuple
from .chromosome import Chromosome
from .fitness_evaluator import FitnessEvaluator
from .params_manager import save_tuned_parameters
from .utils import log_generation_to_csv, calculate_population_diversity


class GeneticOptimizer:
    """Genetic algorithm implementation for parameter tuning."""
    
    def __init__(self, config: Dict):
        """
        Initialize the genetic optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.population_size = config.get("population_size", 30)
        self.generations = config.get("generations", 20)
        self.elitism_count = config.get("elitism_count", 2)
        self.tournament_size = config.get("tournament_size", 5)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.mutation_magnitude = config.get("mutation_magnitude", 0.2)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.save_interval = config.get("save_interval", 5)
        
        self.output_dir = config.get("output_dir", "../tuned_params")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "ga_log.txt")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GeneticOptimizer")
        
        # Initialize the fitness evaluator
        self.evaluator = FitnessEvaluator(config)
        
        # Track total execution time
        self.start_time = time.time()
    
    def run(self) -> Chromosome:
        """
        Run the genetic algorithm optimization process.
        
        Returns:
            Best chromosome found
        """
        self.start_time = time.time()
        self.logger.info(f"Starting genetic optimization with {self.population_size} chromosomes "
                         f"for {self.generations} generations")
        
        # Initialize population
        population = self._initialize_population()
        
        # Track the best chromosome across all generations
        best_overall = None
        best_fitness = float('-inf')
        
        # Run for specified number of generations
        for generation in range(self.generations):
            gen_start_time = time.time()
            
            # Evaluate fitness for entire population
            fitness_scores = self.evaluator.evaluate_population(population)
            
            # Update chromosomes with their fitness scores
            for i, chromosome in enumerate(population):
                chromosome.fitness = fitness_scores[i]
            
            # Find the best chromosome in this generation
            best_gen = max(population, key=lambda x: x.fitness)
            
            # Update overall best if needed
            improvement_found = False
            if best_overall is None or best_gen.fitness > best_fitness:
                best_overall = best_gen.clone()
                best_fitness = best_gen.fitness
                improvement_found = True
                
                # Save new best immediately
                self._save_best_chromosome(best_overall)
                self.logger.info(f"New best chromosome found with fitness {best_fitness:.4f}")
            
            # Calculate statistics for logging
            gen_time = time.time() - gen_start_time
            total_time = time.time() - self.start_time
            avg_fitness = sum(c.fitness for c in population) / len(population)
            min_fitness = min(c.fitness for c in population)
            
            # Calculate standard deviation of fitness
            if len(population) > 1:
                std_dev = statistics.stdev(c.fitness for c in population)
            else:
                std_dev = 0
            
            # Calculate population diversity
            diversity = calculate_population_diversity(population)
            
            # Get win rates for best chromosome
            best_tiger_win_rate, best_goat_win_rate = self._get_win_rates(best_gen)
            
            # Log progress
            self.logger.info(f"Generation {generation + 1}/{self.generations}: "
                            f"Best={best_gen.fitness:.4f}, Avg={avg_fitness:.4f}, "
                            f"Min={min_fitness:.4f}, Diversity={diversity:.4f}, "
                            f"Time={gen_time:.2f}s, Total={total_time:.2f}s")
            
            # Create data for CSV logging
            gen_data = {
                'generation': generation + 1,
                'timestamp': time.time(),
                'best_fitness': best_gen.fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'std_dev': std_dev,
                'elapsed_time': gen_time,
                'total_time': total_time,
                'best_tiger_win_rate': best_tiger_win_rate,
                'best_goat_win_rate': best_goat_win_rate,
                'population_diversity': diversity
            }
            
            # Log to CSV
            log_generation_to_csv(self.output_dir, gen_data)
            
            # Save generation results periodically or when improvement is found
            if (generation + 1) % self.save_interval == 0 or improvement_found:
                self._save_generation_results(population, generation + 1)
            
            # Create next generation (except for the last iteration)
            if generation < self.generations - 1:
                population = self._create_next_generation(population)
        
        # Final logging
        total_time = time.time() - self.start_time
        self.logger.info(f"Optimization complete. Best fitness: {best_fitness:.4f}, "
                        f"Total time: {total_time:.2f}s")
        
        return best_overall
    
    def run_with_time_limit(self, max_time_seconds: float) -> Chromosome:
        """
        Run the genetic algorithm optimization process with a time limit.
        
        Args:
            max_time_seconds: Maximum execution time in seconds
            
        Returns:
            Best chromosome found
        """
        self.start_time = time.time()
        self.logger.info(f"Starting genetic optimization with {self.population_size} chromosomes "
                         f"for up to {max_time_seconds} seconds")
        
        # Initialize population
        population = self._initialize_population()
        
        # Track the best chromosome across all generations
        best_overall = None
        best_fitness = float('-inf')
        
        # Run generations until time limit is reached
        generation = 0
        while time.time() - self.start_time < max_time_seconds and generation < self.generations:
            gen_start_time = time.time()
            
            # Evaluate fitness for entire population
            fitness_scores = self.evaluator.evaluate_population(population)
            
            # Update chromosomes with their fitness scores
            for i, chromosome in enumerate(population):
                chromosome.fitness = fitness_scores[i]
            
            # Find the best chromosome in this generation
            best_gen = max(population, key=lambda x: x.fitness)
            
            # Update overall best if needed
            improvement_found = False
            if best_overall is None or best_gen.fitness > best_fitness:
                best_overall = best_gen.clone()
                best_fitness = best_gen.fitness
                improvement_found = True
                
                # Save new best immediately
                self._save_best_chromosome(best_overall)
                self.logger.info(f"New best chromosome found with fitness {best_fitness:.4f}")
            
            # Calculate statistics for logging
            gen_time = time.time() - gen_start_time
            total_time = time.time() - self.start_time
            avg_fitness = sum(c.fitness for c in population) / len(population)
            min_fitness = min(c.fitness for c in population)
            
            # Calculate standard deviation of fitness
            if len(population) > 1:
                std_dev = statistics.stdev(c.fitness for c in population)
            else:
                std_dev = 0
            
            # Calculate population diversity
            diversity = calculate_population_diversity(population)
            
            # Get win rates for best chromosome
            best_tiger_win_rate, best_goat_win_rate = self._get_win_rates(best_gen)
            
            # Log progress
            self.logger.info(f"Generation {generation + 1}/{self.generations}: "
                            f"Best={best_gen.fitness:.4f}, Avg={avg_fitness:.4f}, "
                            f"Min={min_fitness:.4f}, Diversity={diversity:.4f}, "
                            f"Time={gen_time:.2f}s, Total={total_time:.2f}s")
            
            # Create data for CSV logging
            gen_data = {
                'generation': generation + 1,
                'timestamp': time.time(),
                'best_fitness': best_gen.fitness,
                'avg_fitness': avg_fitness,
                'min_fitness': min_fitness,
                'std_dev': std_dev,
                'elapsed_time': gen_time,
                'total_time': total_time,
                'best_tiger_win_rate': best_tiger_win_rate,
                'best_goat_win_rate': best_goat_win_rate,
                'population_diversity': diversity
            }
            
            # Log to CSV
            log_generation_to_csv(self.output_dir, gen_data)
            
            # Save generation results periodically or when improvement is found
            if (generation + 1) % self.save_interval == 0 or improvement_found:
                self._save_generation_results(population, generation + 1)
            
            # Check if we're about to hit the time limit
            time_elapsed = time.time() - self.start_time
            time_remaining = max_time_seconds - time_elapsed
            
            # If not enough time for another generation, stop
            if time_remaining < gen_time * 1.5:
                self.logger.info(f"Time limit approaching after {generation + 1} generations, stopping.")
                break
            
            # Create next generation
            population = self._create_next_generation(population)
            generation += 1
        
        # Final logging
        total_time = time.time() - self.start_time
        self.logger.info(f"Optimization complete. Best fitness: {best_fitness:.4f}, "
                        f"Total time: {total_time:.2f}s, Generations: {generation + 1}")
        
        return best_overall
    
    def _initialize_population(self) -> List[Chromosome]:
        """
        Initialize a population of random chromosomes.
        
        Returns:
            List of chromosomes
        """
        return [Chromosome(config=self.config) for _ in range(self.population_size)]
    
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
    
    def _save_best_chromosome(self, chromosome: Chromosome):
        """
        Save the best chromosome to file.
        
        Args:
            chromosome: Best chromosome
        """
        best_path = os.path.join(self.output_dir, "best_params.json")
        chromosome.save_to_file(best_path)
        
        # Also save just the genes in a separate file for easier access
        save_tuned_parameters(chromosome.to_dict(), best_path)
        
        # Create a timestamp backup copy for history
        timestamp = int(time.time())
        backup_path = os.path.join(self.output_dir, f"best_params_{timestamp}.json")
        chromosome.save_to_file(backup_path)
    
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
        diversity = calculate_population_diversity(population)
        
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
            # The fitness evaluator caches results by chromosome hash
            fitness = self.evaluator.evaluate_chromosome(chromosome)
            
            # If we can access the raw win rates from evaluator, use them
            if hasattr(self.evaluator, "last_tiger_win_rate") and hasattr(self.evaluator, "last_goat_win_rate"):
                return self.evaluator.last_tiger_win_rate, self.evaluator.last_goat_win_rate
            
            # Otherwise, use a rough estimate
            # Assuming fitness is approximately 50% tiger performance + 50% goat performance
            # This is an approximation based on how fitness_evaluator.py combines scores
            tiger_estimate = min(1.0, max(0.0, fitness * 0.5))
            goat_estimate = min(1.0, max(0.0, fitness * 0.5))
            
            return tiger_estimate, goat_estimate
        except Exception as e:
            self.logger.error(f"Error getting win rates: {e}")
            return 0.0, 0.0 