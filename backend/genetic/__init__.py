"""
Genetic Algorithm package for tuning MinimaxAgent parameters.
"""

from .params_manager import (
    get_tuned_parameters,
    apply_tuned_parameters,
    save_tuned_parameters,
)

__all__ = [
    'get_tuned_parameters',
    'apply_tuned_parameters',
    'save_tuned_parameters',
] 