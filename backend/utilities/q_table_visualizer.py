import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
from utilities.q_table_summarizer import load_q_table, get_metadata, get_output_dir, get_q_tables_dir

def visualize_q_tables():
    """Generate visualizations for the Q-tables"""
    
    # Create output directory
    output_dir = get_output_dir()
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load Q-tables
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
    
    # Generate visualizations
    visualize_training_results(metadata, vis_dir)
    visualize_q_value_distribution(tiger_q_table, goat_q_table, vis_dir)
    visualize_action_distribution(tiger_q_table, goat_q_table, vis_dir)
    visualize_feature_importance(tiger_q_table, goat_q_table, vis_dir)
    
    print(f"Visualizations generated successfully in '{vis_dir}' directory")

def visualize_training_results(metadata, output_dir):
    """Visualize the training results"""
    # Create a pie chart for game outcomes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Game outcome data
    labels = ['Tiger Wins', 'Goat Wins', 'Draws']
    sizes = [
        metadata.get('tiger_wins', 0),
        metadata.get('goat_wins', 0), 
        metadata.get('draws', 0)
    ]
    
    # Calculate percentages
    total = sum(sizes)
    percentages = [f"{s/total*100:.1f}%" for s in sizes]
    
    # Create labels with counts and percentages
    labels = [f"{l} ({s}, {p})" for l, s, p in zip(labels, sizes, percentages)]
    
    # Plot pie chart
    ax.pie(sizes, labels=labels, autopct='', 
           shadow=False, startangle=90, 
           colors=['#FF9999', '#66B2FF', '#99FF99'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title('Game Outcomes after Q-Learning Training')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'game_outcomes.png'), dpi=300)
    plt.close()

def visualize_q_value_distribution(tiger_q_table, goat_q_table, output_dir):
    """Visualize the distribution of Q-values"""
    # Extract Q-values
    tiger_q_values = []
    for actions in tiger_q_table.values():
        tiger_q_values.extend(list(actions.values()))
    
    goat_q_values = []
    for actions in goat_q_table.values():
        goat_q_values.extend(list(actions.values()))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histograms
    bins = np.linspace(-10, 100, 50)
    ax.hist(tiger_q_values, bins=bins, alpha=0.7, label='Tiger Q-values', color='#FF9999')
    ax.hist(goat_q_values, bins=bins, alpha=0.7, label='Goat Q-values', color='#66B2FF')
    
    ax.set_xlabel('Q-value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Q-values')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_value_distribution.png'), dpi=300)
    plt.close()
    
    # Create boxplot for an alternative view
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplot data
    boxplot_data = [
        tiger_q_values,
        goat_q_values
    ]
    
    # Plot boxplot
    box = ax.boxplot(boxplot_data, patch_artist=True, labels=['Tiger', 'Goat'],
                    showfliers=False)  # Hide outliers for better visibility
                    
    # Set box colors
    colors = ['#FF9999', '#66B2FF']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Q-value')
    ax.set_title('Q-value Distribution by Agent')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'q_value_boxplot.png'), dpi=300)
    plt.close()

def visualize_action_distribution(tiger_q_table, goat_q_table, output_dir):
    """Visualize the distribution of actions"""
    # Count actions for tiger
    tiger_actions = Counter()
    for actions in tiger_q_table.values():
        for action in actions:
            tiger_actions[action] += 1
    
    # Count actions for goat
    goat_actions = Counter()
    for actions in goat_q_table.values():
        for action in actions:
            goat_actions[action] += 1
    
    # Create DataFrames
    tiger_df = pd.DataFrame({
        'Action': list(tiger_actions.keys()),
        'Count': list(tiger_actions.values())
    }).sort_values('Count', ascending=False)
    
    goat_df = pd.DataFrame({
        'Action': list(goat_actions.keys()),
        'Count': list(goat_actions.values())
    }).sort_values('Count', ascending=False)
    
    # Create bar plots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(tiger_df['Action'], tiger_df['Count'], color='#FF9999')
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Tiger Actions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tiger_action_distribution.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(goat_df['Action'], goat_df['Count'], color='#66B2FF')
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Goat Actions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goat_action_distribution.png'), dpi=300)
    plt.close()
    
    # Calculate average Q-values per action
    tiger_action_q_values = defaultdict(list)
    for actions in tiger_q_table.values():
        for action, q_value in actions.items():
            tiger_action_q_values[action].append(q_value)
    
    goat_action_q_values = defaultdict(list)
    for actions in goat_q_table.values():
        for action, q_value in actions.items():
            goat_action_q_values[action].append(q_value)
    
    # Calculate average Q-values
    tiger_avg_q = {action: np.mean(values) for action, values in tiger_action_q_values.items()}
    goat_avg_q = {action: np.mean(values) for action, values in goat_action_q_values.items()}
    
    # Create DataFrames
    tiger_q_df = pd.DataFrame({
        'Action': list(tiger_avg_q.keys()),
        'Average Q-Value': list(tiger_avg_q.values())
    }).sort_values('Average Q-Value', ascending=False)
    
    goat_q_df = pd.DataFrame({
        'Action': list(goat_avg_q.keys()),
        'Average Q-Value': list(goat_avg_q.values())
    }).sort_values('Average Q-Value', ascending=False)
    
    # Create bar plots for average Q-values
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(tiger_q_df['Action'], tiger_q_df['Average Q-Value'], color='#FF9999')
    ax.set_xlabel('Action')
    ax.set_ylabel('Average Q-Value')
    ax.set_title('Average Q-Values for Tiger Actions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tiger_action_q_values.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(goat_q_df['Action'], goat_q_df['Average Q-Value'], color='#66B2FF')
    ax.set_xlabel('Action')
    ax.set_ylabel('Average Q-Value')
    ax.set_title('Average Q-Values for Goat Actions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goat_action_q_values.png'), dpi=300)
    plt.close()

def visualize_feature_importance(tiger_q_table, goat_q_table, output_dir):
    """Visualize the importance of different state features"""
    # Feature names
    feature_names = [
        "Goats Captured", "Tigers Trapped", "Tiger Mobility", 
        "Goat Mobility", "Captures Available", "Threats", 
        "Position Score", "Spacing Score", "Edge Score", 
        "Closed Spaces", "Goats on Board", "Phase"
    ]
    
    # Extract feature values for states with high Q-values
    tiger_high_q_states = []
    for state, actions in tiger_q_table.items():
        max_q = max(actions.values()) if actions else 0
        if max_q > 50:  # Only consider states with high Q-values
            tiger_high_q_states.append(state)
    
    goat_high_q_states = []
    for state, actions in goat_q_table.items():
        max_q = max(actions.values()) if actions else 0
        if max_q > 50:  # Only consider states with high Q-values
            goat_high_q_states.append(state)
    
    # Count feature values for tiger
    tiger_feature_counts = [Counter() for _ in range(len(feature_names))]
    for state in tiger_high_q_states:
        for i, value in enumerate(state):
            tiger_feature_counts[i][value] += 1
    
    # Count feature values for goat
    goat_feature_counts = [Counter() for _ in range(len(feature_names))]
    for state in goat_high_q_states:
        for i, value in enumerate(state):
            goat_feature_counts[i][value] += 1
    
    # Create heatmap for feature value distributions
    for i, feature in enumerate(feature_names):
        # Get all possible values for this feature
        tiger_values = set(tiger_feature_counts[i].keys())
        goat_values = set(goat_feature_counts[i].keys())
        all_values = sorted(tiger_values.union(goat_values), key=str)
        
        if all_values:  # Skip empty features
            # Create heatmap data
            tiger_data = [tiger_feature_counts[i].get(v, 0) for v in all_values]
            goat_data = [goat_feature_counts[i].get(v, 0) for v in all_values]
            
            # Normalize to percentages
            tiger_total = sum(tiger_data)
            goat_total = sum(goat_data)
            
            if tiger_total > 0 and goat_total > 0:
                tiger_pct = [100 * count / tiger_total for count in tiger_data]
                goat_pct = [100 * count / goat_total for count in goat_data]
                
                # Create data for heatmap
                heatmap_data = pd.DataFrame({
                    'Tiger': tiger_pct,
                    'Goat': goat_pct
                }, index=[str(v) for v in all_values])
                
                # Create heatmap
                plt.figure(figsize=(10, max(6, len(all_values) * 0.5)))
                sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
                plt.title(f'Distribution of {feature} Values in High-Q States (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'feature_{i+1}_{feature.lower().replace(" ", "_")}.png'), dpi=300)
                plt.close()

if __name__ == "__main__":
    visualize_q_tables() 