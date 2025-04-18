"""
Configuration settings for the simulation module.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

@dataclass
class ParallelRange:
    """Configuration for a parallel execution range."""
    start: Optional[int] = None  # If None, starts from 0
    end: Optional[int] = None    # If None, runs to the end

@dataclass
class MCTSConfigGroup:
    """Configuration group for a set of MCTS parameters to test."""
    rollout_policies: List[str] = field(default_factory=lambda: ["lightweight"])
    iterations: List[int] = field(default_factory=lambda: [None])  # Default to None (use time-based)
    rollout_depths: List[int] = field(default_factory=lambda: [6])
    exploration_weights: List[float] = field(default_factory=lambda: [1.414])  # √2 is optimal for UCB1
    guided_strictness_values: List[float] = field(default_factory=lambda: [0.5])

@dataclass
class MCTSTournamentConfig:
    """Configuration for MCTS tournament."""
    configurations: List[MCTSConfigGroup] = field(default_factory=list)
    max_simulation_time: int = 60  # Maximum time in minutes to run the simulation
    max_time_per_move: int = 10    # Maximum time per move in seconds
    output_dir: str = "simulation_results"
    parallel_ranges: List[ParallelRange] = None  # For parallel execution
    parallel_games: int = None  # Number of parallel games to run per process

    def __post_init__(self):
        # Ensure we have at least one configuration group
        if not self.configurations:
            self.configurations.append(MCTSConfigGroup())

    def get_all_configs(self) -> List[Dict]:
        """
        Generate all MCTS configurations from all configuration groups.
        
        Returns:
            List of all configurations to test
        """
        all_configs = []
        
        for config_group in self.configurations:
            # Generate combinations for this group
            for policy in config_group.rollout_policies:
                for iteration in config_group.iterations:
                    for depth in config_group.rollout_depths:
                        for exploration_weight in config_group.exploration_weights:
                            # Only use multiple guided_strictness values for guided policy
                            if policy == "guided":
                                strictness_values = config_group.guided_strictness_values
                            else:
                                # For non-guided policies, just use the first value in the list
                                strictness_values = [config_group.guided_strictness_values[0]]
                                
                            for guided_strictness in strictness_values:
                                config = {
                                    'algorithm': 'mcts',
                                    'rollout_policy': policy,
                                    'rollout_depth': depth,
                                    'exploration_weight': exploration_weight,
                                    'guided_strictness': guided_strictness,
                                    'max_time_seconds': self.max_time_per_move
                                }
                                
                                # Only add iterations if specified
                                if iteration is not None:
                                    config['iterations'] = iteration
                                
                                all_configs.append(config)
        
        return all_configs
    
    def validate_ranges(self, total_matchups: int) -> List[Dict[str, int]]:
        """
        Validate and process parallel ranges.
        
        Args:
            total_matchups: Total number of matchups to process
            
        Returns:
            List of validated range dictionaries with start and end indices
            
        Raises:
            ValueError: If ranges are invalid
        """
        if not self.parallel_ranges:
            return [{"start": 0, "end": total_matchups}]
            
        # Convert to dictionaries and validate
        validated_ranges = []
        last_end = 0
        
        for i, range_config in enumerate(self.parallel_ranges):
            start = range_config.start if range_config.start is not None else last_end
            end = range_config.end if range_config.end is not None else total_matchups
            
            # Validate range
            if start < 0:
                raise ValueError(f"Range {i}: start index cannot be negative")
            if end > total_matchups:
                raise ValueError(f"Range {i}: end index exceeds total matchups")
            if start >= end:
                raise ValueError(f"Range {i}: start index must be less than end index")
            if start < last_end:
                raise ValueError(f"Range {i}: overlaps with previous range")
                
            validated_ranges.append({"start": start, "end": end})
            last_end = end
            
        # Ensure all matchups are covered
        if last_end < total_matchups:
            validated_ranges.append({"start": last_end, "end": total_matchups})
            
        return validated_ranges

@dataclass
class SimulationConfig:
    """Main configuration for all simulations."""
    sheets_webapp_url: Optional[str] = None  # URL for Google Sheets web app
    sheets_batch_size: int = 100  # Batch size for Google Sheets sync
    mcts_tournament: MCTSTournamentConfig = None
    
    def __post_init__(self):
        if self.mcts_tournament is None:
            self.mcts_tournament = MCTSTournamentConfig()

def get_config_path(config_path: str = "mcts_simulation_config.json") -> str:
    """
    Get the absolute path to the configuration file.
    
    Args:
        config_path: Relative or absolute path to the configuration file
        
    Returns:
        Absolute path to the configuration file
    """
    # Try relative to current directory
    if os.path.isabs(config_path):
        return config_path
        
    # Try relative to current directory
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
        
    # Try relative to simulation directory
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    sim_path = os.path.join(sim_dir, config_path)
    if os.path.exists(sim_path):
        return sim_path
        
    # Default to simulation directory
    return sim_path

def load_config(config_path: str = "mcts_simulation_config.json") -> SimulationConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SimulationConfig object
    """
    config_path = get_config_path(config_path)
    
    if not os.path.exists(config_path):
        # Create default config if it doesn't exist
        print(f"Config file {config_path} not found. Creating default configuration...")
        default_config = SimulationConfig()
        save_config(default_config, config_path)
        print(f"Created default config at {config_path}")
        return default_config
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Get top level configuration fields
    sheets_webapp_url = config_dict.get('sheets_webapp_url')
    sheets_batch_size = config_dict.get('sheets_batch_size', 100)
    
    # Convert nested dictionaries to config objects
    mcts_config = None
    
    if 'mcts_tournament' in config_dict:
        mcts_dict = config_dict['mcts_tournament']
        
        # Handle parallel ranges
        if 'parallel_ranges' in mcts_dict:
            ranges = []
            for r in mcts_dict.get('parallel_ranges', []):
                ranges.append(ParallelRange(
                    start=r.get('start'),
                    end=r.get('end')
                ))
            mcts_dict['parallel_ranges'] = ranges
        
        # Extract max_time_per_move
        max_time_per_move = mcts_dict.get('max_time_per_move', 10)  # Default to 10 seconds
        
        # Handle configuration groups
        if 'configurations' in mcts_dict:
            config_groups = []
            for group in mcts_dict.get('configurations', []):
                # Handle both new and old parameter names for backward compatibility
                exploration_weights = group.get('exploration_weights', None)
                if exploration_weights is None:
                    # If new parameter name not found, try old name and convert to list
                    exploration_weight = group.get('exploration_weight', 1.414)
                    exploration_weights = [exploration_weight]
                
                guided_strictness_values = group.get('guided_strictness_values', None)
                if guided_strictness_values is None:
                    # If new parameter name not found, try old name and convert to list
                    guided_strictness = group.get('guided_strictness', 0.5)
                    guided_strictness_values = [guided_strictness]
                
                # Handle iterations - default to None (time-based) if not specified
                iterations = group.get('iterations', [None])
                
                config_groups.append(MCTSConfigGroup(
                    rollout_policies=group.get('rollout_policies', ["lightweight"]),
                    iterations=iterations,
                    rollout_depths=group.get('rollout_depths', [6]),
                    exploration_weights=exploration_weights,
                    guided_strictness_values=guided_strictness_values
                ))
            mcts_dict['configurations'] = config_groups
            
        # Create tournament config with max_time_per_move
        mcts_config = MCTSTournamentConfig(
            configurations=mcts_dict.get('configurations', []),
            max_simulation_time=mcts_dict.get('max_simulation_time'),
            max_time_per_move=max_time_per_move,
            output_dir=mcts_dict.get('output_dir', 'simulation_results'),
            parallel_ranges=mcts_dict.get('parallel_ranges'),
            parallel_games=mcts_dict.get('parallel_games')
        )
        
    # Create the complete config
    return SimulationConfig(
        sheets_webapp_url=sheets_webapp_url,
        sheets_batch_size=sheets_batch_size,
        mcts_tournament=mcts_config
    )

def save_config(config: SimulationConfig, config_path: str = "mcts_simulation_config.json"):
    """
    Save configuration to a JSON file.
    
    Args:
        config: SimulationConfig object
        config_path: Path to save the configuration file
    """
    config_path = get_config_path(config_path)
    
    config_dict = convert_to_dict(config)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def convert_to_dict(obj):
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for k, v in asdict(obj).items():
            if v is not None:
                result[k] = convert_to_dict(v)
        return result
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items() if v is not None}
    else:
        return obj 