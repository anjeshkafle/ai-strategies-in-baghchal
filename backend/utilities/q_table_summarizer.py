import json
import os
import sys
import math
import pandas as pd
from collections import defaultdict, Counter

# Resolve output directory path
def get_output_dir():
    """Get the output directory path"""
    # Make paths work correctly regardless of where it's run from
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "simulation_results", "q_table_report")

def get_q_tables_dir():
    """Get the q_tables directory path"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "simulation_results", "q_tables")

def load_json_file(file_path):
    """Load a JSON file and return its contents"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_metadata():
    """Load and return the metadata file"""
    metadata_path = os.path.join(get_q_tables_dir(), "metadata.json")
    return load_json_file(metadata_path)

def load_q_table(path):
    """Load a Q-table from a JSON file"""
    data = load_json_file(path)
    if not data:
        return None
    
    # Convert string state keys back to tuples for better analysis
    deserialized = {}
    for state_key, actions in data.items():
        # Using eval to convert string representation of tuple back to actual tuple
        try:
            state_tuple = eval(state_key)
            deserialized[state_tuple] = actions
        except:
            # If there's an error, just use the string key
            deserialized[state_key] = actions
    
    return deserialized

def analyze_q_table(q_table):
    """Analyze a Q-table and return statistics"""
    if not q_table:
        return None
    
    # Basic statistics
    num_states = len(q_table)
    
    # Count actions per state
    actions_per_state = [len(actions) for actions in q_table.values()]
    avg_actions_per_state = sum(actions_per_state) / num_states if num_states > 0 else 0
    max_actions = max(actions_per_state) if actions_per_state else 0
    min_actions = min(actions_per_state) if actions_per_state else 0
    
    # Analyze Q-values
    all_q_values = []
    for actions in q_table.values():
        all_q_values.extend(actions.values())
    
    max_q_value = max(all_q_values) if all_q_values else 0
    min_q_value = min(all_q_values) if all_q_values else 0
    avg_q_value = sum(all_q_values) / len(all_q_values) if all_q_values else 0
    
    # Analyze feature distributions
    feature_distributions = analyze_feature_distributions(q_table)
    
    # Extract top states by Q-value
    top_states = get_top_states(q_table, 5)
    
    # Action distribution
    action_distribution = analyze_action_distribution(q_table)
    
    return {
        "num_states": num_states,
        "avg_actions_per_state": avg_actions_per_state,
        "max_actions": max_actions,
        "min_actions": min_actions,
        "max_q_value": max_q_value,
        "min_q_value": min_q_value,
        "avg_q_value": avg_q_value,
        "feature_distributions": feature_distributions,
        "top_states": top_states,
        "action_distribution": action_distribution
    }

def analyze_feature_distributions(q_table):
    """Analyze the distribution of features in state tuples"""
    feature_values = defaultdict(Counter)
    
    # Feature indices (based on _get_state_features implementations):
    # 0: goats_captured_bucket
    # 1: tigers_trapped_bucket
    # 2: tiger_mobility_bucket
    # 3: goat_mobility_bucket
    # 4: captures_bucket
    # 5: threats_bucket
    # 6: position_bucket
    # 7: spacing_bucket
    # 8: edge_bucket
    # 9: closed_bucket
    # 10: goats_bucket
    # 11: phase
    
    feature_names = [
        "goats_captured", "tigers_trapped", "tiger_mobility", 
        "goat_mobility", "captures", "threats", "position", 
        "spacing", "edge", "closed", "goats", "phase"
    ]
    
    for state in q_table.keys():
        for i, feature in enumerate(state):
            feature_values[i][feature] += 1
    
    distributions = {}
    for i, counter in feature_values.items():
        distributions[feature_names[i]] = dict(counter)
    
    return distributions

def get_top_states(q_table, n=5):
    """Get the top n states with highest Q-values"""
    top_states = []
    
    for state, actions in q_table.items():
        max_q = max(actions.values()) if actions else 0
        best_action = max(actions.items(), key=lambda x: x[1])[0] if actions else None
        
        top_states.append({
            "state": state,
            "best_action": best_action,
            "q_value": max_q
        })
    
    return sorted(top_states, key=lambda x: x["q_value"], reverse=True)[:n]

def analyze_action_distribution(q_table):
    """Analyze the distribution of actions and their average Q-values"""
    action_counts = Counter()
    action_q_sums = defaultdict(float)
    
    for actions in q_table.values():
        for action, q_value in actions.items():
            action_counts[action] += 1
            action_q_sums[action] += q_value
    
    # Calculate average Q-value per action
    action_avg_q = {}
    for action, count in action_counts.items():
        action_avg_q[action] = action_q_sums[action] / count
    
    return {
        "counts": dict(action_counts),
        "avg_q_values": action_avg_q
    }

def create_summary_table(tiger_q_stats, goat_q_stats, metadata):
    """Create a summary table for displaying in the report"""
    summary = {
        "Training Information": {
            "Episodes Completed": metadata.get("completed_episodes", "N/A"),
            "Final Exploration Rate": metadata.get("exploration_rate", "N/A"),
            "Training Time (hours)": round(metadata.get("training_seconds", 0) / 3600, 2),
            "Tiger Wins": metadata.get("tiger_wins", "N/A"),
            "Goat Wins": metadata.get("goat_wins", "N/A"),
            "Draws": metadata.get("draws", "N/A"),
            "Win Rate (Tiger)": f"{(metadata.get('tiger_wins', 0) / (metadata.get('tiger_wins', 0) + metadata.get('goat_wins', 0) + metadata.get('draws', 0)) * 100):.2f}%"
        },
        "Tiger Q-Table": {
            "States": tiger_q_stats["num_states"],
            "Avg. Actions per State": f"{tiger_q_stats['avg_actions_per_state']:.2f}",
            "Max Q-Value": f"{tiger_q_stats['max_q_value']:.2f}",
            "Min Q-Value": f"{tiger_q_stats['min_q_value']:.2f}",
            "Avg Q-Value": f"{tiger_q_stats['avg_q_value']:.2f}"
        },
        "Goat Q-Table": {
            "States": goat_q_stats["num_states"],
            "Avg. Actions per State": f"{goat_q_stats['avg_actions_per_state']:.2f}",
            "Max Q-Value": f"{goat_q_stats['max_q_value']:.2f}",
            "Min Q-Value": f"{goat_q_stats['min_q_value']:.2f}",
            "Avg Q-Value": f"{goat_q_stats['avg_q_value']:.2f}"
        }
    }
    
    return summary

def create_feature_distribution_tables(tiger_q_stats, goat_q_stats):
    """Create tables showing the distribution of state features"""
    tiger_features = tiger_q_stats["feature_distributions"]
    goat_features = goat_q_stats["feature_distributions"]
    
    feature_tables = {}
    
    for feature_name in tiger_features.keys():
        tiger_dist = tiger_features[feature_name]
        goat_dist = goat_features[feature_name]
        
        # Combine all possible values
        all_values = set(list(tiger_dist.keys()) + list(goat_dist.keys()))
        
        feature_table = {
            "Value": [],
            "Tiger States (%)": [],
            "Goat States (%)": []
        }
        
        tiger_total = sum(tiger_dist.values())
        goat_total = sum(goat_dist.values())
        
        # Sort values, but convert to string first to handle mixed types
        for value in sorted(all_values, key=lambda x: str(x)):
            feature_table["Value"].append(value)
            
            tiger_pct = (tiger_dist.get(value, 0) / tiger_total * 100) if tiger_total > 0 else 0
            goat_pct = (goat_dist.get(value, 0) / goat_total * 100) if goat_total > 0 else 0
            
            feature_table["Tiger States (%)"].append(f"{tiger_pct:.1f}%")
            feature_table["Goat States (%)"].append(f"{goat_pct:.1f}%")
        
        feature_tables[feature_name] = feature_table
    
    return feature_tables

def create_action_tables(tiger_q_stats, goat_q_stats):
    """Create tables showing action distribution and Q-values"""
    tiger_actions = tiger_q_stats["action_distribution"]
    goat_actions = goat_q_stats["action_distribution"]
    
    tiger_table = {
        "Action": [],
        "Count": [],
        "% of States": [],
        "Avg Q-Value": []
    }
    
    goat_table = {
        "Action": [],
        "Count": [],
        "% of States": [],
        "Avg Q-Value": []
    }
    
    tiger_total = tiger_q_stats["num_states"]
    goat_total = goat_q_stats["num_states"]
    
    for action, count in sorted(tiger_actions["counts"].items(), key=lambda x: x[1], reverse=True):
        tiger_table["Action"].append(action)
        tiger_table["Count"].append(count)
        tiger_table["% of States"].append(f"{count/tiger_total*100:.1f}%")
        tiger_table["Avg Q-Value"].append(f"{tiger_actions['avg_q_values'][action]:.2f}")
    
    for action, count in sorted(goat_actions["counts"].items(), key=lambda x: x[1], reverse=True):
        goat_table["Action"].append(action)
        goat_table["Count"].append(count)
        goat_table["% of States"].append(f"{count/goat_total*100:.1f}%")
        goat_table["Avg Q-Value"].append(f"{goat_actions['avg_q_values'][action]:.2f}")
    
    return {"tiger": tiger_table, "goat": goat_table}

def create_best_states_table(tiger_q_stats, goat_q_stats):
    """Create a table showing the top states by Q-value"""
    tiger_top = tiger_q_stats["top_states"]
    goat_top = goat_q_stats["top_states"]
    
    def format_state_features(state):
        # Feature names in order
        feature_names = [
            "Goats Captured", "Tigers Trapped", "Tiger Mobility", 
            "Goat Mobility", "Captures Available", "Threats", 
            "Position Score", "Spacing Score", "Edge Score", 
            "Closed Spaces", "Goats on Board", "Phase"
        ]
        
        return ", ".join([f"{name}: {value}" for name, value in zip(feature_names, state)])
    
    tiger_table = {
        "Rank": list(range(1, len(tiger_top) + 1)),
        "State Features": [format_state_features(s["state"]) for s in tiger_top],
        "Best Action": [s["best_action"] for s in tiger_top],
        "Q-Value": [f"{s['q_value']:.2f}" for s in tiger_top]
    }
    
    goat_table = {
        "Rank": list(range(1, len(goat_top) + 1)),
        "State Features": [format_state_features(s["state"]) for s in goat_top],
        "Best Action": [s["best_action"] for s in goat_top],
        "Q-Value": [f"{s['q_value']:.2f}" for s in goat_top]
    }
    
    return {"tiger": tiger_table, "goat": goat_table}

def generate_report():
    """Generate a summary report of the Q-tables"""
    
    # Load final Q-tables for tiger and goat
    q_tables_dir = get_q_tables_dir()
    tiger_q_table = load_q_table(os.path.join(q_tables_dir, "tiger_q_final.json"))
    goat_q_table = load_q_table(os.path.join(q_tables_dir, "goat_q_final.json"))
    
    if not tiger_q_table or not goat_q_table:
        print("Failed to load Q-tables")
        return
        
    # Load metadata
    metadata = get_metadata()
    if not metadata:
        print("Failed to load metadata")
        return
    
    # Analyze Q-tables
    tiger_q_stats = analyze_q_table(tiger_q_table)
    goat_q_stats = analyze_q_table(goat_q_table)
    
    # Create summary tables
    summary_table = create_summary_table(tiger_q_stats, goat_q_stats, metadata)
    feature_tables = create_feature_distribution_tables(tiger_q_stats, goat_q_stats)
    action_tables = create_action_tables(tiger_q_stats, goat_q_stats)
    best_states_tables = create_best_states_table(tiger_q_stats, goat_q_stats)
    
    # Create a directory for the report
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a markdown file for the summary report
    with open(os.path.join(output_dir, "q_table_summary.md"), "w") as f:
        f.write("# Q-Learning Agent Analysis for Bagh Chal\n\n")
        
        # Training information
        f.write("## Training Information\n\n")
        training_df = pd.DataFrame.from_dict(summary_table["Training Information"], orient='index', columns=["Value"])
        f.write(training_df.to_markdown())
        f.write("\n\n")
        
        # Q-table statistics
        f.write("## Q-Table Statistics\n\n")
        f.write("### Tiger Q-Table\n\n")
        tiger_df = pd.DataFrame.from_dict(summary_table["Tiger Q-Table"], orient='index', columns=["Value"])
        f.write(tiger_df.to_markdown())
        f.write("\n\n")
        
        f.write("### Goat Q-Table\n\n")
        goat_df = pd.DataFrame.from_dict(summary_table["Goat Q-Table"], orient='index', columns=["Value"])
        f.write(goat_df.to_markdown())
        f.write("\n\n")
        
        # Action distribution
        f.write("## Action Distribution\n\n")
        f.write("### Tiger Actions\n\n")
        tiger_actions_df = pd.DataFrame(action_tables["tiger"])
        f.write(tiger_actions_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Goat Actions\n\n")
        goat_actions_df = pd.DataFrame(action_tables["goat"])
        f.write(goat_actions_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Best states
        f.write("## Top States by Q-Value\n\n")
        f.write("### Tiger Best States\n\n")
        tiger_states_df = pd.DataFrame(best_states_tables["tiger"])
        f.write(tiger_states_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Goat Best States\n\n")
        goat_states_df = pd.DataFrame(best_states_tables["goat"])
        f.write(goat_states_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Feature distributions (selected most important)
        f.write("## Selected Feature Distributions\n\n")
        important_features = ["goats_captured", "tigers_trapped", "tiger_mobility", "goat_mobility", "phase"]
        
        for feature in important_features:
            f.write(f"### {feature.replace('_', ' ').title()}\n\n")
            feature_df = pd.DataFrame(feature_tables[feature])
            f.write(feature_df.to_markdown(index=False))
            f.write("\n\n")
            
        # Add a note about the full analysis
        f.write("## Note\n\n")
        f.write("This summary provides an overview of the Q-learning agents trained for Bagh Chal. ")
        f.write("The complete Q-tables are too large to include in this report but are available in the project repository. ")
        f.write("The full data can be accessed at [Google Drive Link] where researchers can analyze the complete Q-tables.\n\n")
        f.write("**State Features Explanation:**\n\n")
        f.write("* **Goats Captured**: Number of goats captured by tigers (0-5)\n")
        f.write("* **Tigers Trapped**: Number of tigers with no legal moves (0-4)\n")
        f.write("* **Tiger/Goat Mobility**: Categorized mobility level based on available moves\n")
        f.write("* **Captures Available**: Immediate capture opportunities for tigers\n")
        f.write("* **Threats**: Potential future capture setups\n")
        f.write("* **Position/Spacing/Edge Scores**: Strategic position evaluations\n")
        f.write("* **Closed Spaces**: Regions where goats have blocked tiger movement\n")
        f.write("* **Goats on Board**: Number of goats currently on the board\n")
        f.write("* **Phase**: Game phase (PLACEMENT or MOVEMENT)\n")
    
    # Create CSV files for the tabular data for easier inclusion in the report
    # Summary table
    training_df = pd.DataFrame.from_dict(summary_table["Training Information"], orient='index', columns=["Value"])
    training_df.to_csv(os.path.join(output_dir, "training_info.csv"))
    
    # Q-table statistics
    tiger_df = pd.DataFrame.from_dict(summary_table["Tiger Q-Table"], orient='index', columns=["Value"])
    tiger_df.to_csv(os.path.join(output_dir, "tiger_stats.csv"))
    
    goat_df = pd.DataFrame.from_dict(summary_table["Goat Q-Table"], orient='index', columns=["Value"])
    goat_df.to_csv(os.path.join(output_dir, "goat_stats.csv"))
    
    # Action distribution
    pd.DataFrame(action_tables["tiger"]).to_csv(os.path.join(output_dir, "tiger_actions.csv"), index=False)
    pd.DataFrame(action_tables["goat"]).to_csv(os.path.join(output_dir, "goat_actions.csv"), index=False)
    
    # Best states
    pd.DataFrame(best_states_tables["tiger"]).to_csv(os.path.join(output_dir, "tiger_best_states.csv"), index=False)
    pd.DataFrame(best_states_tables["goat"]).to_csv(os.path.join(output_dir, "goat_best_states.csv"), index=False)
    
    # Create README file
    create_readme(output_dir)
    
    print(f"Report generated successfully in the '{output_dir}' directory")
    print(f"- Markdown report: {os.path.join(output_dir, 'q_table_summary.md')}")
    print(f"- CSV files for data tables are also available in the same directory")

def create_readme(output_dir):
    """Create a README file for the q_table_report directory"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Q-Learning Agent Analysis for Bagh Chal\n\n")
        f.write("This directory contains analyses, summaries, and visualizations of the Q-learning agents trained for the Bagh Chal game. The data here can be included in research papers, thesis documents, or presentations to explain the behavior and performance of the reinforcement learning agents.\n\n")
        
        f.write("## Contents\n\n")
        f.write("### Summary Document\n\n")
        f.write("- **q_table_summary.md**: Comprehensive markdown document summarizing the Q-tables\n")
        f.write("  - Training information and outcomes\n")
        f.write("  - Q-table statistics\n")
        f.write("  - Action distributions\n")
        f.write("  - Top states by Q-value\n")
        f.write("  - Feature value distributions\n")
        f.write("  - Explanations of state features\n\n")
        
        f.write("### CSV Files for Easy Import\n\n")
        f.write("- **training_info.csv**: Basic training parameters and results\n")
        f.write("- **tiger_stats.csv** / **goat_stats.csv**: Summary statistics for tiger and goat Q-tables\n")
        f.write("- **tiger_actions.csv** / **goat_actions.csv**: Action distribution data\n")
        f.write("- **tiger_best_states.csv** / **goat_best_states.csv**: Information about the top-performing states\n\n")
        
        f.write("### Visualizations\n\n")
        f.write("The `visualizations/` directory contains various plots and charts:\n\n")
        f.write("- **game_outcomes.png**: Pie chart of game outcomes after training\n")
        f.write("- **q_value_distribution.png** / **q_value_boxplot.png**: Distribution of Q-values\n")
        f.write("- **tiger_action_distribution.png** / **goat_action_distribution.png**: Action frequency plots\n")
        f.write("- **tiger_action_q_values.png** / **goat_action_q_values.png**: Average Q-value per action\n")
        f.write("- **feature_*.png**: Feature value distributions in high-Q states\n\n")
        
        f.write("## How to Use These Resources\n\n")
        f.write("### For MSc Thesis/Report\n\n")
        f.write("1. Include the `q_table_summary.md` content in your appendix\n")
        f.write("2. Import the CSV files into your analysis tools or use them to create tables\n")
        f.write("3. Select the most relevant visualizations for your methodology and results sections\n")
        f.write("4. Direct readers to your Google Drive or repository for the complete Q-tables\n\n")
        
        f.write("### To Generate Your Own Analysis\n\n")
        f.write("This analysis was generated using two Python scripts in the backend/utilities directory:\n\n")
        f.write("1. **q_table_summarizer.py**: Creates the summary tables and markdown document\n")
        f.write("   ```\n")
        f.write("   cd backend\n")
        f.write("   python -m utilities.q_table_summarizer\n")
        f.write("   ```\n\n")
        
        f.write("2. **q_table_visualizer.py**: Generates all the visualizations\n")
        f.write("   ```\n")
        f.write("   cd backend\n")
        f.write("   python -m utilities.q_table_visualizer\n")
        f.write("   ```\n\n")
        
        f.write("You can modify these scripts to generate additional analyses or focus on specific aspects of the Q-tables.\n\n")
        
        f.write("## Understanding the Q-Tables\n\n")
        f.write("The Q-tables are structured as:\n\n")
        f.write("- **Key**: State features tuple with 12 elements representing the game state\n")
        f.write("- **Value**: Dictionary mapping abstract actions to Q-values\n\n")
        
        f.write("### State Features (in order)\n\n")
        f.write("1. Goats Captured (0-5)\n")
        f.write("2. Tigers Trapped (0-4)\n")
        f.write("3. Tiger Mobility (Low/Med/High)\n")
        f.write("4. Goat Mobility (Low/Med/High)\n")
        f.write("5. Captures Available (0, 1, 2+)\n")
        f.write("6. Threats (None/Low/High)\n")
        f.write("7. Position Score (Low/Med/High)\n")
        f.write("8. Spacing Score (Low/Med/High)\n")
        f.write("9. Edge Score (Low/Med/High)\n")
        f.write("10. Closed Spaces (None/Low/High)\n")
        f.write("11. Goats on Board (Few/Mid/Many)\n")
        f.write("12. Phase (PLACEMENT/MOVEMENT)\n\n")
        
        f.write("### Tiger Actions\n\n")
        f.write("- Capture_Goat: Capture a goat piece\n")
        f.write("- Improve_Position: Move to a better position\n")
        f.write("- Safe_Move: Make a move that doesn't put the tiger at risk\n")
        f.write("- Improve_Spacing: Optimize spacing between tigers\n")
        f.write("- Increase_Mobility: Move to increase tiger mobility\n")
        f.write("- Setup_Threat: Create a future capture opportunity\n\n")
        
        f.write("### Goat Actions\n\n")
        f.write("- Block_Immediate_Capture: Prevent a tiger from capturing\n")
        f.write("- Safe_Placement/Safe_Move: Place/Move to a non-threatened position\n")
        f.write("- Improve_Edge_Position: Optimize edge position\n")
        f.write("- Reduce_Tiger_Mobility: Place/Move to limit tiger mobility\n")
        f.write("- Escape_Threat: Move away from threatened position\n")
        f.write("- Contribute_To_Trap: Help trap tigers\n\n")
        
        f.write("## Complete Q-Table Access\n\n")
        f.write("The complete Q-tables are too large to include directly. They can be accessed at:\n")
        f.write("[INSERT YOUR GOOGLE DRIVE OR REPOSITORY LINK HERE]")

if __name__ == "__main__":
    generate_report() 