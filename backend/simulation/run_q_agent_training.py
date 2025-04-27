import os
import json
import time
import argparse
import signal
import sys
from models.q_agent import QLearningAgent
from models.game_state import GameState

# Global variable to track if we're interrupted
interrupted = False

def signal_handler(sig, frame):
    """Handle keyboard interrupts gracefully"""
    global interrupted
    print("\nInterrupted by user, finishing current episode and saving...")
    interrupted = True

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def ensure_dir(directory):
    """Make sure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_path):
    """Load training configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def get_latest_episode(save_path, prefix):
    """Find the latest episode number from saved files"""
    if not os.path.exists(save_path):
        return 0
        
    latest_episode = 0
    for filename in os.listdir(save_path):
        if filename.startswith(prefix) and filename.endswith('.json'):
            try:
                # Extract episode number from filenames like "tiger_q_100.json"
                parts = filename.split('_')
                episode = int(parts[-1].split('.')[0])
                latest_episode = max(latest_episode, episode)
            except (ValueError, IndexError):
                pass
    
    return latest_episode

def save_metadata(save_path, metadata):
    """Save training metadata to track progress"""
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

def load_metadata(save_path):
    """Load training metadata"""
    metadata_path = os.path.join(save_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {
        "completed_episodes": 0, 
        "exploration_rate": 1.0,
        "tiger_wins": 0,
        "goat_wins": 0,
        "draws": 0,
        "training_seconds": 0
    }

def run_training(config):
    """Run Q-agent training with the given configuration"""
    # Extract config values with defaults
    episodes = config.get("episodes", 10000)
    save_interval = config.get("save_interval", 100)
    backup_interval = config.get("backup_interval", 10)  # New parameter for more frequent backups
    save_path = config.get("save_path", "models/q_tables")
    discount_factor = config.get("discount_factor", 0.95)
    initial_exploration_rate = config.get("initial_exploration_rate", 1.0)
    min_exploration_rate = config.get("min_exploration_rate", 0.05)
    exploration_decay = config.get("exploration_decay", 0.99)
    max_time_seconds = config.get("max_time_seconds", None)
    seed = config.get("seed", None)
    
    # Determine training mode based on presence of coach settings
    coach_settings = config.get("coach_settings", None)
    if coach_settings:
        training_mode = "coach"
        role_to_train = coach_settings.get("role_to_train", "TIGER")
        coach_type = coach_settings.get("coach_type", "minimax")
        coach_params = coach_settings.get("coach_params", {"max_depth": 3})
    else:
        training_mode = "self_play"
        role_to_train = None
        coach_type = None
        coach_params = None
    
    # Ensure save directory exists
    ensure_dir(save_path)
    
    # Check for existing training metadata
    metadata = load_metadata(save_path)
    completed_episodes = metadata["completed_episodes"]
    exploration_rate = metadata["exploration_rate"]
    training_seconds = metadata["training_seconds"]
    
    print(f"Resuming training from episode {completed_episodes+1}")
    
    # Initialize Q-agent
    q_agent = QLearningAgent(
        discount_factor=discount_factor,
        initial_exploration_rate=exploration_rate,  # Use saved exploration rate
        min_exploration_rate=min_exploration_rate,
        exploration_decay=exploration_decay,
        seed=seed,
        auto_load=False,  # Don't auto-load here, we'll handle loading manually
        tables_path=save_path  # Use the save_path as the tables_path
    )
    
    # Load existing Q-tables if available
    if completed_episodes > 0:
        # Find latest files
        if training_mode == "self_play":
            # In self-play, both agents should have tables
            tiger_q = os.path.join(save_path, f"tiger_q_{completed_episodes}.json")
            tiger_v = os.path.join(save_path, f"tiger_v_{completed_episodes}.json")
            goat_q = os.path.join(save_path, f"goat_q_{completed_episodes}.json")
            goat_v = os.path.join(save_path, f"goat_v_{completed_episodes}.json")
            
            # Check if files exist
            if not all(os.path.exists(f) for f in [tiger_q, tiger_v, goat_q, goat_v]):
                print("Warning: Some Q-table files missing. Looking for latest complete set...")
                
                # Find latest episode with all files
                for ep in range(completed_episodes, 0, -save_interval):
                    tiger_q = os.path.join(save_path, f"tiger_q_{ep}.json")
                    tiger_v = os.path.join(save_path, f"tiger_v_{ep}.json")
                    goat_q = os.path.join(save_path, f"goat_q_{ep}.json")
                    goat_v = os.path.join(save_path, f"goat_v_{ep}.json")
                    
                    if all(os.path.exists(f) for f in [tiger_q, tiger_v, goat_q, goat_v]):
                        completed_episodes = ep
                        print(f"Found complete set at episode {ep}")
                        break
            
            # Load tables
            q_agent.load_tables(tiger_q, tiger_v, goat_q, goat_v)
            
        else:  # Coach training
            # In coach training, only the learning agent has tables
            if role_to_train == "TIGER":
                tiger_q = os.path.join(save_path, f"tiger_q_{completed_episodes}.json")
                tiger_v = os.path.join(save_path, f"tiger_v_{completed_episodes}.json")
                
                # Check if files exist
                if not all(os.path.exists(f) for f in [tiger_q, tiger_v]):
                    print("Warning: Some Q-table files missing. Looking for latest complete set...")
                    
                    # Find latest episode with all files
                    for ep in range(completed_episodes, 0, -save_interval):
                        tiger_q = os.path.join(save_path, f"tiger_q_{ep}.json")
                        tiger_v = os.path.join(save_path, f"tiger_v_{ep}.json")
                        
                        if all(os.path.exists(f) for f in [tiger_q, tiger_v]):
                            completed_episodes = ep
                            print(f"Found complete set at episode {ep}")
                            break
                
                # Load tables
                q_agent.load_tables(tiger_q, tiger_v, None, None)
                
            else:  # GOAT
                goat_q = os.path.join(save_path, f"goat_q_{completed_episodes}.json")
                goat_v = os.path.join(save_path, f"goat_v_{completed_episodes}.json")
                
                # Check if files exist
                if not all(os.path.exists(f) for f in [goat_q, goat_v]):
                    print("Warning: Some Q-table files missing. Looking for latest complete set...")
                    
                    # Find latest episode with all files
                    for ep in range(completed_episodes, 0, -save_interval):
                        goat_q = os.path.join(save_path, f"goat_q_{ep}.json")
                        goat_v = os.path.join(save_path, f"goat_v_{ep}.json")
                        
                        if all(os.path.exists(f) for f in [goat_q, goat_v]):
                            completed_episodes = ep
                            print(f"Found complete set at episode {ep}")
                            break
                
                # Load tables
                q_agent.load_tables(None, None, goat_q, goat_v)
    
    # Initialize trackers
    start_time = time.time()
    total_training_time = training_seconds
    
    # Custom training loop to handle interruptions and statistics
    remaining_episodes = episodes - completed_episodes
    current_episode = completed_episodes + 1
    
    # Create Q-learning update callbacks for tracking statistics
    tiger_wins = metadata.get("tiger_wins", 0)
    goat_wins = metadata.get("goat_wins", 0)
    draws = metadata.get("draws", 0)
    
    print(f"Starting training in {training_mode} mode for {remaining_episodes} more episodes...")
    if training_mode == "coach":
        print(f"Training {role_to_train} agent against {coach_type} coach")
    print(f"Current statistics - Tiger wins: {tiger_wins}, Goat wins: {goat_wins}, Draws: {draws}")
    
    try:
        # Set up training loop manually to track wins/losses
        if training_mode == "self_play":
            # Training through self-play
            epsilon = q_agent.initial_exploration_rate
            
            for episode in range(current_episode, episodes + 1):
                # Check for interruption or time limit
                if interrupted:
                    print("Training interrupted by user")
                    break
                
                if max_time_seconds and (time.time() - start_time + total_training_time) > max_time_seconds:
                    print(f"Training stopped due to time limit")
                    break
                
                # Play one episode
                state = GameState()
                game_length = 0
                
                # Track visited states for threefold repetition
                visited_states = {}  # Format: {state_hash: count}
                
                # Helper function to get a hash representation of the game state
                def get_state_hash(game_state):
                    # Only track repetition during movement phase
                    if game_state.phase == "MOVEMENT":
                        # Convert board to a string representation
                        board_str = ""
                        for row in game_state.board:
                            for cell in row:
                                if cell is None:
                                    board_str += "_"
                                elif cell["type"] == "TIGER":
                                    board_str += "T"
                                else:
                                    board_str += "G"
                        
                        # Include turn in the hash
                        return f"{board_str}_{game_state.turn}"
                    else:
                        # During placement phase, include goats_placed to ensure uniqueness
                        return f"PLACEMENT_{game_state.goats_placed}_{game_state.turn}"
                
                # Play until game end
                while not state.is_terminal():
                    game_length += 1
                    
                    # Check for threefold repetition in movement phase
                    if state.phase == "MOVEMENT":
                        state_hash = get_state_hash(state)
                        visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                        if visited_states[state_hash] >= 3:
                            # Game ends in a draw due to threefold repetition
                            draws += 1
                            break
                    
                    if state.turn == "TIGER":
                        # Tiger's turn
                        prev_state = state.clone()
                        abstract_action = q_agent.tiger_agent._choose_action(state, epsilon)
                        if abstract_action is None:
                            break
                        concrete_move = q_agent.tiger_agent._get_concrete_move(state, abstract_action)
                        if concrete_move is None:
                            break
                        
                        # Apply move
                        state.apply_move(concrete_move)
                        
                        # Get immediate reward and update
                        reward = q_agent.tiger_agent._get_reward(prev_state, abstract_action, state)
                        q_agent.tiger_agent.update_q_table(prev_state, abstract_action, reward, state)
                        
                    else:  # GOAT's turn
                        # Goat's turn
                        prev_state = state.clone()
                        abstract_action = q_agent.goat_agent._choose_action(state, epsilon)
                        if abstract_action is None:
                            break
                        concrete_move = q_agent.goat_agent._get_concrete_move(state, abstract_action)
                        if concrete_move is None:
                            break
                        
                        # Apply move
                        state.apply_move(concrete_move)
                        
                        # Get immediate reward and update
                        reward = q_agent.goat_agent._get_reward(prev_state, abstract_action, state)
                        q_agent.goat_agent.update_q_table(prev_state, abstract_action, reward, state)
                
                # Update statistics
                winner = state.get_winner()
                if winner == "TIGER":
                    tiger_wins += 1
                elif winner == "GOAT":
                    goat_wins += 1
                else:
                    draws += 1
                
                # Report progress periodically
                if episode % 10 == 0 or episode == 1:
                    elapsed_time = time.time() - start_time + total_training_time
                    print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.3f}, Game Length: {game_length}, " + 
                          f"Winner: {winner}, T/G/D: {tiger_wins}/{goat_wins}/{draws}, Elapsed: {elapsed_time:.1f}s")
                
                # Backup to main files frequently
                if episode % backup_interval == 0 or interrupted:
                    # Save to the main 4 files (overwriting)
                    q_agent.save_tables(
                        f"{save_path}/tiger_q_final.json",
                        f"{save_path}/tiger_v_final.json",
                        f"{save_path}/goat_q_final.json",
                        f"{save_path}/goat_v_final.json"
                    )
                    
                    # Update metadata
                    current_time = time.time() - start_time + total_training_time
                    metadata = {
                        "completed_episodes": episode,
                        "exploration_rate": epsilon,
                        "tiger_wins": tiger_wins,
                        "goat_wins": goat_wins,
                        "draws": draws,
                        "training_seconds": current_time
                    }
                    save_metadata(save_path, metadata)
                    
                    if episode % backup_interval == 0:
                        print(f"Backed up progress at episode {episode}")
                
                # Create snapshots only if save_interval is positive
                if save_interval > 0 and episode % save_interval == 0:
                    # Save the numbered snapshot files
                    q_agent.save_tables(
                        f"{save_path}/tiger_q_{episode}.json",
                        f"{save_path}/tiger_v_{episode}.json",
                        f"{save_path}/goat_q_{episode}.json",
                        f"{save_path}/goat_v_{episode}.json"
                    )
                    print(f"Created snapshot at episode {episode}")
                
                # Decay epsilon
                epsilon = max(q_agent.min_exploration_rate, epsilon * q_agent.exploration_decay)
        
        else:  # Coach training
            # Create coach and train with it
            epsilon = q_agent.initial_exploration_rate
            
            # Create coach agent
            coach = q_agent._create_coach_agent(coach_type, coach_params)
            if coach is None:
                print(f"Failed to create coach agent of type {coach_type}")
                return
            
            for episode in range(current_episode, episodes + 1):
                # Check for interruption or time limit
                if interrupted:
                    print("Training interrupted by user")
                    break
                
                if max_time_seconds and (time.time() - start_time + total_training_time) > max_time_seconds:
                    print(f"Training stopped due to time limit")
                    break
                
                # Play one episode
                state = GameState()
                game_length = 0
                
                # Track visited states for threefold repetition
                visited_states = {}  # Format: {state_hash: count}
                
                # Play until game end
                while not state.is_terminal():
                    game_length += 1
                    
                    # Check for threefold repetition in movement phase
                    if state.phase == "MOVEMENT":
                        state_hash = get_state_hash(state)
                        visited_states[state_hash] = visited_states.get(state_hash, 0) + 1
                        if visited_states[state_hash] >= 3:
                            # Game ends in a draw due to threefold repetition
                            draws += 1
                            break
                    
                    # Determine if the current player is the learner or the coach
                    is_learner_turn = (state.turn == role_to_train)
                    
                    if is_learner_turn:
                        # Learner's turn
                        prev_state = state.clone()
                        
                        # Choose an action with exploration
                        if role_to_train == "TIGER":
                            abstract_action = q_agent.tiger_agent._choose_action(state, epsilon)
                            if abstract_action is None:
                                break
                            concrete_move = q_agent.tiger_agent._get_concrete_move(state, abstract_action)
                        else:  # GOAT
                            abstract_action = q_agent.goat_agent._choose_action(state, epsilon)
                            if abstract_action is None:
                                break
                            concrete_move = q_agent.goat_agent._get_concrete_move(state, abstract_action)
                            
                        if concrete_move is None:
                            break
                            
                        # Apply move
                        state.apply_move(concrete_move)
                        
                        # Get reward and update Q-table
                        if role_to_train == "TIGER":
                            reward = q_agent.tiger_agent._get_reward(prev_state, abstract_action, state)
                            q_agent.tiger_agent.update_q_table(prev_state, abstract_action, reward, state)
                        else:  # GOAT
                            reward = q_agent.goat_agent._get_reward(prev_state, abstract_action, state)
                            q_agent.goat_agent.update_q_table(prev_state, abstract_action, reward, state)
                        
                    else:
                        # Coach's turn
                        coach_move = coach.get_move(state)
                        if coach_move is None:
                            break
                        
                        # Apply coach's move
                        state.apply_move(coach_move)
                
                # Update statistics
                winner = state.get_winner()
                if winner == "TIGER":
                    tiger_wins += 1
                elif winner == "GOAT":
                    goat_wins += 1
                else:
                    draws += 1
                
                # Report progress periodically
                if episode % 10 == 0 or episode == 1:
                    elapsed_time = time.time() - start_time + total_training_time
                    print(f"Episode {episode}/{episodes}, Epsilon: {epsilon:.3f}, Game Length: {game_length}, " + 
                          f"Winner: {winner}, T/G/D: {tiger_wins}/{goat_wins}/{draws}, Elapsed: {elapsed_time:.1f}s")
                
                # Backup to main files frequently
                if episode % backup_interval == 0 or interrupted:
                    # Save to the main files based on training role
                    if role_to_train == "TIGER":
                        q_agent.save_tables(
                            f"{save_path}/tiger_q_final.json",
                            f"{save_path}/tiger_v_final.json",
                            None,
                            None
                        )
                    else:  # GOAT
                        q_agent.save_tables(
                            None,
                            None,
                            f"{save_path}/goat_q_final.json",
                            f"{save_path}/goat_v_final.json"
                        )
                    
                    # Update metadata
                    current_time = time.time() - start_time + total_training_time
                    metadata = {
                        "completed_episodes": episode,
                        "exploration_rate": epsilon,
                        "tiger_wins": tiger_wins,
                        "goat_wins": goat_wins,
                        "draws": draws,
                        "training_seconds": current_time
                    }
                    save_metadata(save_path, metadata)
                    
                    if episode % backup_interval == 0:
                        print(f"Backed up progress at episode {episode}")
                
                # Create snapshots only if save_interval is positive
                if save_interval > 0 and episode % save_interval == 0:
                    # Save the numbered snapshot files based on training role
                    if role_to_train == "TIGER":
                        q_agent.save_tables(
                            f"{save_path}/tiger_q_{episode}.json",
                            f"{save_path}/tiger_v_{episode}.json",
                            None,
                            None
                        )
                    else:  # GOAT
                        q_agent.save_tables(
                            None,
                            None,
                            f"{save_path}/goat_q_{episode}.json",
                            f"{save_path}/goat_v_{episode}.json"
                        )
                    print(f"Created snapshot at episode {episode}")
                
                # Decay epsilon
                epsilon = max(q_agent.min_exploration_rate, epsilon * q_agent.exploration_decay)
    
    finally:
        # Final save
        episode = min(episodes, current_episode)
        
        # No need to create another snapshot - we'll just ensure final files are up to date
        if training_mode == "self_play":
            q_agent.save_tables(
                f"{save_path}/tiger_q_final.json",
                f"{save_path}/tiger_v_final.json",
                f"{save_path}/goat_q_final.json",
                f"{save_path}/goat_v_final.json"
            )
        else:  # Coach training
            if role_to_train == "TIGER":
                q_agent.save_tables(
                    f"{save_path}/tiger_q_final.json",
                    f"{save_path}/tiger_v_final.json",
                    None,
                    None
                )
            else:  # GOAT
                q_agent.save_tables(
                    None,
                    None,
                    f"{save_path}/goat_q_final.json",
                    f"{save_path}/goat_v_final.json"
                )
        
        # Final metadata update
        current_time = time.time() - start_time + total_training_time
        metadata = {
            "completed_episodes": episode,
            "exploration_rate": epsilon,
            "tiger_wins": tiger_wins,
            "goat_wins": goat_wins,
            "draws": draws,
            "training_seconds": current_time
        }
        save_metadata(save_path, metadata)
        
        print("\nTraining complete!")
        print(f"Completed {episode} episodes.")
        print(f"Final statistics - Tiger wins: {tiger_wins}, Goat wins: {goat_wins}, Draws: {draws}")
        print(f"Total training time: {current_time:.1f} seconds")
        
        return q_agent

def main():
    """Main function to parse arguments and run training"""
    parser = argparse.ArgumentParser(description='Train a Q-learning agent for Bagh Chal')
    parser.add_argument('--config', type=str, default='backend/simulation/q_training_config.json',
                        help='Path to training configuration JSON file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    run_training(config)

if __name__ == "__main__":
    main() 