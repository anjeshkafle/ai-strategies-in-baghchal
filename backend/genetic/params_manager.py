"""
Manages parameter loading and application for the MinimaxAgent.
This module exposes tuned parameters to the main codebase.
"""
import os
import json
from typing import Dict, Any, Optional
import logging


# Default path for the best parameters file
DEFAULT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "tuned_params", "best_params.json")


def get_tuned_parameters(
    params_path: str = DEFAULT_PARAMS_PATH,
    fallback_to_defaults: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get tuned parameters from the best params file.
    
    Args:
        params_path: Path to the parameters file
        fallback_to_defaults: If True, return None when file doesn't exist
        
    Returns:
        Dictionary of parameters or None if not found
    """
    if not os.path.exists(params_path):
        if fallback_to_defaults:
            return None
        raise FileNotFoundError(f"Tuned parameters file not found: {params_path}")
    
    try:
        with open(params_path, 'r') as f:
            data = json.load(f)
        
        # Return just the genes from the chromosome
        return data.get("genes", data)
    except (json.JSONDecodeError, KeyError) as e:
        if fallback_to_defaults:
            return None
        raise ValueError(f"Error loading tuned parameters: {e}")


def apply_tuned_parameters(agent, params: Dict[str, Any] = None):
    """
    Apply tuned parameters to a MinimaxAgent instance.
    
    Args:
        agent: MinimaxAgent instance to modify
        params: Parameter dictionary or None to load from file
    """
    if params is None:
        params = get_tuned_parameters()
        if params is None:
            return  # No parameters found, keep defaults
    
    # Apply weight parameters
    weight_params = [
        'mobility_weight_placement',
        'mobility_weight_movement',
        'base_capture_value',
        'capture_speed_weight',
        'dispersion_weight',
        'edge_weight',
        'closed_spaces_weight'
    ]
    
    for param in weight_params:
        if param in params:
            if param == 'capture_speed_weight':
                # Special case: this is multiplied by max_depth in the agent
                value = params[param] / agent.max_depth
            else:
                value = params[param]
            setattr(agent, param, value)
    
    # Store factor parameters - these are applied dynamically during evaluation
    # but we store them for reference
    agent.tuned_factors = {
        param: params[param] 
        for param in [
            'closed_space_weight_factor',
            'position_weight_factor', 
            'edge_weight_factor',
            'spacing_weight_factor'
        ] 
        if param in params
    }
    
    # Store equilibrium points - these are applied dynamically during evaluation
    agent.tuned_equilibrium = {
        param: params[param]
        for param in [
            'position_equilibrium_early',
            'position_equilibrium_late',
            'spacing_equilibrium_early',
            'spacing_equilibrium_late',
            'edge_equilibrium_early',
            'edge_equilibrium_mid',
            'edge_equilibrium_late'
        ]
        if param in params
    }
    
    # Flag that the agent is using tuned parameters
    agent._using_tuned_params = True


def save_tuned_parameters(params: Dict[str, Any], output_path: str = DEFAULT_PARAMS_PATH):
    """
    Save tuned parameters to file.
    
    Args:
        params: Parameter dictionary
        output_path: Path to save the parameters
    """
    # Create directory if it doesn't exist - with error handling
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating directory for tuned parameters: {e}")
        raise
    
    # Save parameters with error handling
    try:
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving tuned parameters to {output_path}: {e}")
        raise 