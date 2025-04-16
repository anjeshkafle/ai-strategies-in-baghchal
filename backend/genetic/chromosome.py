"""
Chromosome representation for genetic algorithm tuning of MinimaxAgent parameters.
"""
import json
import random
import copy
import time
from typing import Dict, Any, List, Tuple, Optional


class Chromosome:
    """
    Represents a chromosome in the genetic algorithm.
    A chromosome contains genes that encode parameter values for the MinimaxAgent.
    """
    
    def __init__(self, genes: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, fitness: Optional[float] = None):
        """
        Initialize a chromosome with genes and optional fitness.
        
        Args:
            genes: Dictionary of genes (parameter name -> value)
            config: Configuration dictionary for initialization
            fitness: Optional fitness value
        """
        self.fitness = fitness
        
        if genes is not None:
            self.genes = genes
        elif config is not None:
            self.genes = self._initialize_genes_from_config(config)
        else:
            self.genes = {}
    
    def _initialize_genes_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize genes from the configuration."""
        genes = {}
        
        # Initialize parameters from ranges
        for param, range_values in config["parameter_ranges"].items():
            min_val, max_val = range_values
            genes[param] = random.uniform(min_val, max_val)
        
        # Initialize equilibrium points
        for param, range_values in config["equilibrium_ranges"].items():
            min_val, max_val = range_values
            genes[param] = random.uniform(min_val, max_val)
        
        return genes
    
    def mutate(self, mutation_rate: float, mutation_magnitude: float):
        """
        Apply mutation to genes based on mutation rate and magnitude.
        
        Args:
            mutation_rate: Probability of mutating each gene
            mutation_magnitude: Scale of mutation (relative to parameter range)
        """
        # Define parameter ranges - hardcoded default values
        parameter_ranges = {
            "mobility_weight_placement": [150, 350],
            "mobility_weight_movement": [200, 500],
            "base_capture_value": [2500, 4000],
            "capture_speed_weight": [35, 75],
            "dispersion_weight": [75, 200],
            "edge_weight": [200, 450],
            "closed_spaces_weight": [800, 1300],
            "closed_space_weight_factor": [0.8, 2.0],
            "position_weight_factor": [0.8, 1.5],
            "edge_weight_factor": [0.8, 1.8],
            "spacing_weight_factor": [0.8, 1.5]
        }
        
        # Define equilibrium ranges - hardcoded default values
        equilibrium_ranges = {
            "position_equilibrium_early": [0.4, 0.6],
            "position_equilibrium_late": [0.25, 0.45],
            "spacing_equilibrium_early": [0.4, 0.6],
            "spacing_equilibrium_late": [0.25, 0.45],
            "edge_equilibrium_early": [0.8, 1.0],
            "edge_equilibrium_mid": [0.7, 0.9],
            "edge_equilibrium_late": [0.08, 0.25]
        }
        
        for param in self.genes:
            # Determine if this gene should mutate
            if random.random() < mutation_rate:
                # Get the appropriate range
                if param in parameter_ranges:
                    min_val, max_val = parameter_ranges[param]
                elif param in equilibrium_ranges:
                    min_val, max_val = equilibrium_ranges[param]
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
        # Create two empty children with empty genes
        child1 = cls(genes={})
        child2 = cls(genes={})
        
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
        """
        Create a deep copy of this chromosome.
        
        Returns:
            A new Chromosome instance with the same genes and fitness
        """
        # Create a new chromosome with the same genes
        clone = Chromosome(genes=copy.deepcopy(self.genes), fitness=self.fitness)
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
        chromosome = cls(genes=data["genes"])
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
    
    @classmethod
    def create_random(cls, config: Dict[str, Any]) -> 'Chromosome':
        """
        Create a new chromosome with random gene values based on the configuration.
        
        Args:
            config: Configuration dictionary containing parameter ranges
            
        Returns:
            A new Chromosome instance with random genes
        """
        # Get parameter ranges from config
        parameter_ranges = config.get("parameter_ranges", {})
        equilibrium_ranges = config.get("equilibrium_ranges", {})
        
        # Create random genes
        genes = {}
        
        # Initialize from parameter ranges
        for param, range_values in parameter_ranges.items():
            if isinstance(range_values, list) and len(range_values) == 2:
                min_val, max_val = range_values
                # Generate a random value within the range
                genes[param] = min_val + random.random() * (max_val - min_val)
        
        # Initialize from equilibrium ranges
        for param, range_values in equilibrium_ranges.items():
            if isinstance(range_values, list) and len(range_values) == 2:
                min_val, max_val = range_values
                # Generate a random value within the range
                genes[param] = min_val + random.random() * (max_val - min_val)
        
        # Create and return the new chromosome
        return cls(genes=genes, fitness=None) 