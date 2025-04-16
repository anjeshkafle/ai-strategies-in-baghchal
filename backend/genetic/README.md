# Genetic Algorithm for Minimax Agent Tuning

This module provides tools to optimize the heuristic parameters of the MinimaxAgent using genetic algorithms.

## Quick Start

To run the genetic algorithm optimization:

```bash
python tune_minimax.py
```

The results will be saved in the `backend/tuned_params` directory.

## Configuration

All configurations are managed through the `ga_config.json` file. You can modify the following parameters:

### Core GA Parameters

- `population_size`: Number of chromosomes in each generation (default: 30)
- `generations`: Total number of generations to evolve (default: 20)
- `elitism_count`: Number of best chromosomes to preserve unchanged (default: 2)
- `tournament_size`: Number of chromosomes to compare in tournament selection (default: 5)
- `mutation_rate`: Probability of a gene mutating (default: 0.1)
- `mutation_magnitude`: Scale of mutations when they occur (default: 0.2)
- `crossover_rate`: Probability of crossover between parent chromosomes (default: 0.7)
- `games_per_evaluation`: Number of games to play per evaluation (default: 10)
- `parallel_processes`: Number of parallel processes for game playing (default: cpu_count-1)
  - Set to 0 to automatically use available CPU cores minus 1
  - Set to a specific number to limit process count
- `save_interval`: Generation interval for saving results (default: 5)
- `output_dir`: Directory where results are saved (default: "../tuned_params")

### Parameter Ranges

Each parameter has a defined range in the config file:

```json
"parameter_ranges": {
  "mobility_weight_placement": [100, 500],
  "mobility_weight_movement": [200, 600],
  ...
}
```

You can modify these ranges to extend or narrow the search space.

## Parallelization & Performance

The system uses Python's multiprocessing for true parallel execution:

- Each chromosome is evaluated by playing multiple games
- Games are distributed across available CPU cores
- Setting `parallel_processes` to 0 automatically uses `cpu_count - 1`
- Each game runs in its own process for maximum CPU utilization

The parallelization architecture ensures:

1. Maximum CPU core utilization
2. Windows & Mac compatibility
3. Resilience to game execution errors
4. Fallback to sequential execution if parallel processing fails

## Output Files

The system generates the following files in the output directory:

- `best_params.json`: Best parameters found during optimization
- `best_params_[timestamp].json`: History of best parameters at different points
- `generation_[N].json`: Detailed data for each saved generation
- `ga_log.txt`: Log file with detailed messages
- `ga_progress.csv`: CSV with progress data for analysis
- `fitness_plot.png`: Graph of fitness progression (generated at the end)
- `parameter_evolution.png`: Graph of parameter evolution (generated at the end)
- `optimization_report.txt`: Summary report of the optimization process

## Using Tuned Parameters

After optimization, you can use the tuned parameters in your MinimaxAgent:

```python
from models.minimax_agent import MinimaxAgent

# Create a MinimaxAgent with tuned parameters
agent = MinimaxAgent(useTunedParams=True)
```

## Advanced Features

### Interruption & Resuming

The system saves results after each generation and immediately saves new best parameters when found. To resume an interrupted run, you would need to modify the code to load the last generation.

### Analyzing Results

The CSV file `ga_progress.csv` contains detailed metrics for each generation, including:

- Fitness scores (best, average, min)
- Win rates (tiger, goat)
- Population diversity
- Execution time

Use this data to analyze the optimization process and parameter sensitivity.

## Requirements

- Python 3.6+
- NumPy (for statistical calculations)
- Matplotlib (for plotting results)
- Pandas (for advanced data analysis, optional)

## Customization

To change how parameters are evaluated, modify the `FitnessEvaluator` class in `fitness_evaluator.py`. The current implementation:

- Plays both tiger and goat sides
- Rewards higher win rates
- Rewards faster wins and delayed losses
- Balances both tiger and goat performance
