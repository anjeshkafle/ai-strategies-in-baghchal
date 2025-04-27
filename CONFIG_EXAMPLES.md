# Configuration Examples

This project uses various JSON configuration files to control different aspects of the AI simulation, training, and analysis. Since these config files often contain environment-specific settings or values that shouldn't be committed to version control, we use `.json.example` files as templates.

## How to Use These Example Files

1. Copy any `.json.example` file to its corresponding `.json` file
2. Modify the settings as needed for your environment
3. The actual `.json` files are ignored by git (listed in `.gitignore`)

Example:

```bash
cp backend/simulation/q_training_config.json.example backend/simulation/q_training_config.json
# Then edit q_training_config.json with your specific settings
```

## Understanding the Format

Each example file includes all required parameters with reasonable default values, plus a `_documentation` section that explains what each parameter does. The `_documentation` field should be removed from your working copy.

## Available Configuration Files

### AI Agent Training

- **backend/simulation/q_training_config.json**: Controls Q-learning agent training parameters
- **backend/genetic/ga_config.json**: Controls genetic algorithm optimization settings

### Simulation & Analysis

- **backend/simulation/mcts_simulation_config.json**: MCTS tournament simulation settings
- **backend/simulation/mcts_analysis_config.json**: MCTS performance analysis settings
- **backend/simulation/main_competition_config.json**: Competition between different AI agents
- **backend/simulation/main_analysis_config.json**: Overall analysis settings
- **backend/simulation/genetic_analysis_config.json**: Genetic algorithm performance analysis

## Notes on Sensitive Values

Some configuration files may contain sensitive values (like API URLs). In the example files, these have been replaced with placeholder values. Make sure to enter your actual values when creating your working copies.
