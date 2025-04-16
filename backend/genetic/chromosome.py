"""
Chromosome representation for genetic algorithm tuning of MinimaxAgent parameters.
"""
import json
import random
import copy
from typing import Dict, Any, List, Tuple


class Chromosome:
    """Represents a set of parameter values for the MinimaxAgent."""
    
    def __init__(self, config: Dict = None, random_init: bool = True):
        """
        Initialize a chromosome with either random values or specific values.
        
        Args:
            config: Configuration dict with parameter_ranges
            random_init: If True, initialize with random values within ranges
        """
        self.config = config
        self.genes = {}
        self.fitness = 0.0
        
        if random_init and config:
            self._initialize_random()
    
    def _initialize_random(self):
        """Initialize chromosome with random values within the specified ranges."""
        # Initialize parameters from ranges
        for param, range_values in self.config["parameter_ranges"].items():
            min_val, max_val = range_values
            self.genes[param] = random.uniform(min_val, max_val)
        
        # Initialize equilibrium points
        for param, range_values in self.config["equilibrium_ranges"].items():
            min_val, max_val = range_values
            self.genes[param] = random.uniform(min_val, max_val)
    
    def mutate(self, mutation_rate: float, mutation_magnitude: float):
        """
        Apply mutation to genes based on mutation rate and magnitude.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_magnitude: Scale of mutation (relative to parameter range)
        """
        for param in self.genes:
            # Determine if this gene should mutate
            if random.random() < mutation_rate:
                # Get the appropriate range
                if param in self.config["parameter_ranges"]:
                    min_val, max_val = self.config["parameter_ranges"][param]
                elif param in self.config["equilibrium_ranges"]:
                    min_val, max_val = self.config["equilibrium_ranges"][param]
                else:
                    continue
                
                # Calculate mutation amount
                range_size = max_val - min_val
                mutation_amount = random.gauss(0, mutation_magnitude * range_size)
                
                # Apply mutation and clip to valid range
                self.genes[param] += mutation_amount
                self.genes[param] = max(min_val, min(max_val, self.genes[param]))
    
    @classmethod
    def crossover(cls, parent1: 'Chromosome', parent2: 'Chromosome') -> Tuple['Chromosome', 'Chromosome']:
        """
        Perform uniform crossover between two parent chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two child chromosomes
        """
        # Create two empty children with the same config
        child1 = cls(config=parent1.config, random_init=False)
        child2 = cls(config=parent1.config, random_init=False)
        
        # Copy genes from parents using uniform crossover
        for param in parent1.genes:
            if random.random() < 0.5:
                child1.genes[param] = parent1.genes[param]
                child2.genes[param] = parent2.genes[param]
            else:
                child1.genes[param] = parent2.genes[param]
                child2.genes[param] = parent1.genes[param]
        
        return child1, child2
    
    def clone(self) -> 'Chromosome':
        """Create a deep copy of this chromosome."""
        clone = Chromosome(config=self.config, random_init=False)
        clone.genes = copy.deepcopy(self.genes)
        clone.fitness = self.fitness
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chromosome to dictionary for serialization."""
        return {
            "genes": self.genes,
            "fitness": self.fitness
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Dict) -> 'Chromosome':
        """Create chromosome from dictionary data."""
        chromosome = cls(config=config, random_init=False)
        chromosome.genes = data["genes"]
        chromosome.fitness = data["fitness"]
        return chromosome
    
    def save_to_file(self, filepath: str):
        """Save chromosome to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str, config: Dict) -> 'Chromosome':
        """Load chromosome from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, config)
    
    def __str__(self):
        """String representation showing genes and fitness."""
        return f"Fitness: {self.fitness:.4f}, Genes: {self.genes}" 