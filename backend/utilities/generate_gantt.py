import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# Define the data
data = [
    {'ID': '1.1', 'Task': 'Initial Research & Project Setup', 'Start': '2025-01-27', 'End': '2025-02-02', 'Duration': 7, 'Dependencies': '', 'Phase': 'Phase 1: Foundation & Setup'},
    {'ID': '1.2', 'Task': 'Develop Basic UI (Board, Pieces, Human Input)', 'Start': '2025-02-03', 'End': '2025-02-11', 'Duration': 9, 'Dependencies': '1.1', 'Phase': 'Phase 1: Foundation & Setup'},
    {'ID': '1.3', 'Task': 'Develop Backend Engine & Core Game Logic', 'Start': '2025-02-05', 'End': '2025-02-18', 'Duration': 14, 'Dependencies': '1.1', 'Phase': 'Phase 1: Foundation & Setup'},
    {'ID': '1.4', 'Task': 'Integrate UI and Backend, Basic Game Flow', 'Start': '2025-02-14', 'End': '2025-02-21', 'Duration': 8, 'Dependencies': '1.2, 1.3', 'Phase': 'Phase 1: Foundation & Setup'},
    {'ID': '2.1', 'Task': 'Implement Core Minimax Algorithm & Alpha-Beta Pruning', 'Start': '2025-02-19', 'End': '2025-02-28', 'Duration': 10, 'Dependencies': '1.3', 'Phase': 'Phase 2: Minimax Agent Development'},
    {'ID': '2.2', 'Task': 'Develop Baseline Heuristic Evaluation Function', 'Start': '2025-02-26', 'End': '2025-03-07', 'Duration': 10, 'Dependencies': '2.1', 'Phase': 'Phase 2: Minimax Agent Development'},
    {'ID': '2.3', 'Task': 'Implement Move Ordering & Iterative Deepening', 'Start': '2025-03-05', 'End': '2025-03-11', 'Duration': 7, 'Dependencies': '2.1, 2.2', 'Phase': 'Phase 2: Minimax Agent Development'},
    {'ID': '2.4', 'Task': 'Initial Testing & Debugging of Minimax Agent', 'Start': '2025-03-10', 'End': '2025-03-14', 'Duration': 5, 'Dependencies': '2.3', 'Phase': 'Phase 2: Minimax Agent Development'},
    {'ID': '3.1', 'Task': 'Implement Core MCTS Framework (UCT, Node Structure)', 'Start': '2025-03-12', 'End': '2025-03-21', 'Duration': 10, 'Dependencies': '1.3', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '3.2', 'Task': 'Develop Random Rollout Policy', 'Start': '2025-03-19', 'End': '2025-03-24', 'Duration': 6, 'Dependencies': '3.1', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '3.3', 'Task': 'Develop Lightweight Heuristic Rollout Policy', 'Start': '2025-03-21', 'End': '2025-03-31', 'Duration': 11, 'Dependencies': '3.1', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '3.4', 'Task': 'Develop Guided (Minimax-based) Rollout Policy', 'Start': '2025-03-27', 'End': '2025-04-04', 'Duration': 9, 'Dependencies': '3.1, 2.2', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '3.5', 'Task': 'Implement Win Rate Predictor & Tree Reuse', 'Start': '2025-04-01', 'End': '2025-04-07', 'Duration': 7, 'Dependencies': '3.1', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '3.6', 'Task': 'Initial Testing & Debugging of MCTS Agent', 'Start': '2025-04-03', 'End': '2025-04-09', 'Duration': 7, 'Dependencies': '3.2, 3.3, 3.4, 3.5', 'Phase': 'Phase 3: MCTS Agent Development'},
    {'ID': '4.1', 'Task': 'Implement Genetic Algorithm Framework (Real-coded)', 'Start': '2025-03-28', 'End': '2025-04-04', 'Duration': 8, 'Dependencies': '2.2', 'Phase': 'Phase 4: Minimax Optimization (GA Tuning)'},
    {'ID': '4.2', 'Task': 'Develop Fitness Evaluation (Parallel Minimax Games)', 'Start': '2025-04-02', 'End': '2025-04-08', 'Duration': 7, 'Dependencies': '4.1, 2.4', 'Phase': 'Phase 4: Minimax Optimization (GA Tuning)'},
    {'ID': '4.3', 'Task': 'Execute GA Tuning Experiments', 'Start': '2025-04-07', 'End': '2025-04-18', 'Duration': 12, 'Dependencies': '4.2', 'Phase': 'Phase 4: Minimax Optimization (GA Tuning)'},
    {'ID': '4.4', 'Task': 'Analyze GA Results & Integrate Tuned Parameters', 'Start': '2025-04-18', 'End': '2025-04-21', 'Duration': 4, 'Dependencies': '4.3', 'Phase': 'Phase 4: Minimax Optimization (GA Tuning)'},
    {'ID': '5.1', 'Task': 'Design MCTS Configuration Tournament', 'Start': '2025-04-08', 'End': '2025-04-11', 'Duration': 4, 'Dependencies': '3.6', 'Phase': 'Phase 5: MCTS Optimization (Tournament)'},
    {'ID': '5.2', 'Task': 'Implement Tournament Automation & Logging', 'Start': '2025-04-10', 'End': '2025-04-16', 'Duration': 7, 'Dependencies': '5.1', 'Phase': 'Phase 5: MCTS Optimization (Tournament)'},
    {'ID': '5.3', 'Task': 'Execute MCTS Configuration Tournament', 'Start': '2025-04-14', 'End': '2025-04-22', 'Duration': 9, 'Dependencies': '5.2', 'Phase': 'Phase 5: MCTS Optimization (Tournament)'},
    {'ID': '5.4', 'Task': 'Analyze MCTS Tournament Results & Identify Top Configs', 'Start': '2025-04-21', 'End': '2025-04-24', 'Duration': 4, 'Dependencies': '5.3', 'Phase': 'Phase 5: MCTS Optimization (Tournament)'},
    {'ID': '6.1', 'Task': 'Design Main Tournament (Tuned Minimax vs Elite MCTS)', 'Start': '2025-04-18', 'End': '2025-04-21', 'Duration': 4, 'Dependencies': '4.4, 5.4', 'Phase': 'Phase 6: Main Comparative Evaluation'},
    {'ID': '6.2', 'Task': 'Execute Main Tournament', 'Start': '2025-04-21', 'End': '2025-04-25', 'Duration': 5, 'Dependencies': '6.1, 5.2', 'Phase': 'Phase 6: Main Comparative Evaluation'},
    {'ID': '6.3', 'Task': 'Comprehensive Data Analysis & Statistical Validation', 'Start': '2025-04-24', 'End': '2025-04-27', 'Duration': 4, 'Dependencies': '6.2', 'Phase': 'Phase 6: Main Comparative Evaluation'},
    {'ID': '7.1', 'Task': 'Write Final Project Report', 'Start': '2025-04-21', 'End': '2025-04-27', 'Duration': 7, 'Dependencies': '4.4, 5.4, 6.3', 'Phase': 'Phase 7: Final Reporting & Submission'},
    {'ID': '7.2', 'Task': 'Final Review, Formatting & Submission Preparation', 'Start': '2025-04-26', 'End': '2025-04-28', 'Duration': 3, 'Dependencies': '7.1', 'Phase': 'Phase 7: Final Reporting & Submission'},
    {'ID': '7.3', 'Task': 'Project Submission', 'Start': '2025-04-28', 'End': '2025-04-28', 'Duration': 1, 'Dependencies': '7.2', 'Phase': 'Phase 7: Final Reporting & Submission'},
    {'ID': '8.1', 'Task': 'Implement Proof-of-Concept Q-Learning Agent', 'Start': '2025-04-22', 'End': '2025-04-26', 'Duration': 5, 'Dependencies': '1.3', 'Phase': 'Phase 8: Ancillary Tasks'}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert dates to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Define colors for each phase
phase_colors = {
    'Phase 1: Foundation & Setup': '#4287f5',                        # Blue
    'Phase 2: Minimax Agent Development': '#42f5a7',                 # Green
    'Phase 3: MCTS Agent Development': '#f5a142',                    # Orange
    'Phase 4: Minimax Optimization (GA Tuning)': '#f542cb',          # Pink
    'Phase 5: MCTS Optimization (Tournament)': '#b042f5',            # Purple
    'Phase 6: Main Comparative Evaluation': '#f54242',               # Red
    'Phase 7: Final Reporting & Submission': '#f5f542',              # Yellow
    'Phase 8: Ancillary Tasks': '#42f5e3'                            # Cyan
}

# Get unique phases for the legend
unique_phases = df['Phase'].unique()

# A4 paper dimensions in inches (portrait orientation)
a4_width = 8.27
a4_height = 11.69

# Create the figure and axis - using A4 portrait dimensions
fig, ax = plt.subplots(figsize=(a4_width, a4_height))

# Set the axis labels
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Task', fontsize=10)

# Format the x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))

# Sort tasks by start date and ID for better organization
df = df.sort_values(['Start', 'ID'])

# Create a task ID to Y position mapping
task_positions = {}
for i, task in enumerate(df.iterrows()):
    task_positions[task[1]['ID']] = i

# Create enhanced y-axis labels with IDs
labels = [f"{row['ID']}: {row['Task']}" for _, row in df.iterrows()]
yticks = range(len(labels))
ax.set_yticks(yticks)
ax.set_yticklabels(labels, fontsize=8)

# Correctly balance spacing to use full width
plt.subplots_adjust(left=0.30, right=0.95)  # Adjusted to use more horizontal space

# Set y-axis limits for better spacing
ax.set_ylim(-1, len(labels)) 

# Store bar end positions for drawing dependencies
task_bars = {}

# Add the task bars with appropriate colors
for i, task in df.iterrows():
    start_date = task['Start']
    end_date = task['End'] + timedelta(days=1)  # Add 1 day to include the end date
    duration = (end_date - start_date).days
    
    y_pos = task_positions[task['ID']]
    
    # Get the color based on the phase
    color = phase_colors[task['Phase']]
    
    # Plot the task bar with reduced height to create more spacing
    bar = ax.barh(y_pos, duration, left=start_date, height=0.5, 
            color=color, alpha=0.8, edgecolor='black')
    
    # Store bar positions for drawing dependencies
    task_bars[task['ID']] = {
        'start': start_date,
        'end': end_date,
        'y': y_pos
    }
    
    # Add duration to the bar
    text_x = start_date + timedelta(days=duration / 2)
    ax.text(text_x, y_pos, f"{task['ID']} ({task['Duration']}d)", 
            ha='center', va='center', color='black', fontweight='bold', fontsize=8)

# Add dependency arrows
for i, task in df.iterrows():
    if pd.notna(task['Dependencies']) and task['Dependencies'] != '':
        # Get list of dependencies
        deps = [dep.strip() for dep in task['Dependencies'].split(',')]
        
        for dep in deps:
            if dep in task_bars and task['ID'] in task_bars:
                # Get positions
                start_y = task_bars[dep]['y']
                start_x = task_bars[dep]['end']
                
                end_y = task_bars[task['ID']]['y']
                end_x = task_bars[task['ID']]['start']
                
                # Draw arrow with a slight curve
                arrow_props = dict(arrowstyle='->', color='gray', linewidth=1.0, 
                                connectionstyle='arc3,rad=0.1', alpha=0.7)
                
                # Add the dependency arrow
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), 
                           arrowprops=arrow_props)

# Create custom legend for phases
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=phase_colors[phase], alpha=0.8) 
                  for phase in unique_phases]

# Add dependency arrow to legend
arrow_patch = mpatches.Patch(color='gray', label='Task Dependency')
legend_elements.append(arrow_patch)

# Add legend with phases and dependency explanation
# Use figure coordinates for precise placement
legend = fig.legend(legend_elements, 
          [*unique_phases, 'Task Dependency'], 
          loc='upper center', 
          # Adjust vertical position to have normal margin with x-axis (moved up)
          bbox_to_anchor=(0.5, 0.06),  # Slight adjustment for better spacing
          bbox_transform=fig.transFigure,  # Use figure coordinates
          ncol=3, 
          fontsize=8,
          frameon=True,  # Add frame for visibility
          handlelength=1.0,  # Make handles smaller
          columnspacing=1.0)  # Tighter column spacing

# Format the x-axis for better readability
plt.gcf().autofmt_xdate(rotation=45)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ensure date range is appropriate and fully visible
start_date = min(df['Start']) - timedelta(days=5)
end_date = max(df['End']) + timedelta(days=5)
ax.set_xlim(start_date, end_date)

# Create padding at the bottom for the legend - adjusted for closer placement
plt.subplots_adjust(left=0.30, right=0.95, bottom=0.15, top=0.97)  # Reduced bottom margin from 0.18

# Adjust figure size slightly to ensure consistent margins
fig.set_size_inches(a4_width, a4_height)

# Save the chart as PNG and PDF - use bbox_extra_artists to include legend in save
plt.savefig('bagh_chal_gantt_chart.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])
plt.savefig('bagh_chal_gantt_chart.pdf', bbox_inches='tight', bbox_extra_artists=[legend])

# Show the chart
plt.show()

print("Gantt chart has been generated as 'bagh_chal_gantt_chart.png' and 'bagh_chal_gantt_chart.pdf'") 