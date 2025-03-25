#!/usr/bin/env python3
import os
import json
import math
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import threading
from models.game_state import GameState

class WinRateEntry:
    """
    Represents an entry in the win rate table, tracking outcomes and statistics.
    """
    
    def __init__(self, win_count=0, loss_count=0, draw_count=0):
        """
        Initialize a new win rate entry.
        
        Args:
            win_count: Initial number of wins (tiger)
            loss_count: Initial number of losses (goat)
            draw_count: Initial number of draws
        """
        self.win_count = win_count
        self.loss_count = loss_count
        self.draw_count = draw_count
        self.total_samples = win_count + loss_count + draw_count
        
    def update(self, outcome: float, weight: int = 1) -> None:
        """
        Update the entry with a new game outcome.
        
        Args:
            outcome: Game outcome (1.0 for Tiger win, 0.0 for Goat win, 0.5 for Draw)
            weight: How many samples this outcome represents
        """
        if outcome == 1.0:
            self.update_tiger_win(weight)
        elif outcome == 0.0:
            self.update_goat_win(weight)
        else:
            self.update_draw(weight)
            
    def update_tiger_win(self, weight: int = 1) -> None:
        """
        Update the entry with a tiger win.
        
        Args:
            weight: How many samples this outcome represents
        """
        self.win_count += weight
        self.total_samples += weight
    
    def update_goat_win(self, weight: int = 1) -> None:
        """
        Update the entry with a goat win.
        
        Args:
            weight: How many samples this outcome represents
        """
        self.loss_count += weight
        self.total_samples += weight
    
    def update_draw(self, weight: int = 1) -> None:
        """
        Update the entry with a draw.
        
        Args:
            weight: How many samples this outcome represents
        """
        self.draw_count += weight
        self.total_samples += weight

    @property
    def win_rate(self) -> float:
        """
        Calculate the win rate for this entry.
        
        Returns:
            Win rate from tiger's perspective (0.0-1.0)
        """
        if self.total_samples == 0:
            return 0.5  # Default prior when no data
            
        # Calculate win rate with a small prior to avoid extreme values
        prior_strength = 1.0
        prior_value = 0.5
        
        # Adjusted win rate with prior
        return (self.win_count + prior_strength * prior_value) / (self.total_samples + prior_strength)
    
    @property
    def confidence(self) -> float:
        """
        Calculate the confidence level for this entry.
        
        Returns:
            Confidence value (0.0-1.0) based on sample count
        """
        # Confidence grows with more samples but saturates
        # E.g., 0 samples -> 0.0 confidence
        #      10 samples -> ~0.63 confidence
        #      30 samples -> ~0.87 confidence
        #     100 samples -> ~0.99 confidence
        return 1.0 - math.exp(-self.total_samples / 15.0)
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """
        Calculate 95% confidence interval for the win rate.
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.total_samples == 0:
            return (0.0, 1.0)  # Maximum uncertainty
            
        # Wilson score interval calculation
        z = 1.96  # 95% confidence
        p = self.win_rate
        n = self.total_samples
        
        denominator = 1 + z*z/n
        center = (p + z*z/(2*n)) / denominator
        interval = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denominator
        
        lower = max(0.0, center - interval)
        upper = min(1.0, center + interval)
        
        return (lower, upper)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary for serialization.
        
        Returns:
            Dictionary representation of the entry
        """
        return {
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "draw_count": self.draw_count,
            "total_samples": self.total_samples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WinRateEntry':
        """
        Create entry from dictionary (deserialization).
        
        Args:
            data: Dictionary with entry data
            
        Returns:
            New WinRateEntry instance
        """
        return cls(
            win_count=data.get("win_count", 0),
            loss_count=data.get("loss_count", 0),
            draw_count=data.get("draw_count", 0)
        )


class FeatureExtractor:
    """
    Extracts and processes features from game states for win rate prediction.
    """
    
    # Define feature names and ranges for reference
    FEATURE_NAMES = [
        'goats_placed',           # Number of goats on the board [0-20]
        'goats_captured',         # Number of goats captured [0-5]
        'effective_closed_spaces', # Closed spaces - captured goats [0, 1, 2+]
        'movable_tigers',         # Number of tigers that can move [1-2, 3-4]
        'threatened_goats',       # Goats that can be captured next move [0, 1, 2+]
        'tiger_dispersion',       # Spatial distribution of tigers [Low, Medium, High]
        'goat_edge_preference',   # Goats' preference for edge positions [Low, High]
        'turn'                    # Whose turn (0=GOAT, 1=TIGER)
    ]
    
    FEATURE_RANGES = {
        'goats_placed': list(range(21)),         # 0-20
        'goats_captured': list(range(5)),        # 0-4
        'effective_closed_spaces': [0, 1, 2],    # 0, 1, 2+ (binned values)
        'movable_tigers': [1, 2],                # 1-2, 3-4 (bin boundaries)
        'threatened_goats': [0, 1, 2],           # 0, 1, 2+ (binned values)
        'tiger_dispersion': [0, 1, 2],           # Low, Medium, High
        'goat_edge_preference': [0, 1],          # Low, High
        'turn': [0, 1]                           # 0=GOAT, 1=TIGER
    }
    
    @classmethod
    def _count_closed_spaces(cls, state: GameState) -> int:
        """
        Count the number of closed spaces (spaces where tigers cannot move)
        by importing from the minimax agent.
        
        Args:
            state: The game state
            
        Returns:
            Number of closed spaces
        """
        try:
            from models.minimax_agent import MinimaxAgent
            return MinimaxAgent.count_closed_triangles(state)
        except (ImportError, AttributeError):
            # Fallback if the function can't be imported
            return 0
    
    @classmethod
    def _count_movable_tigers(cls, state: GameState) -> int:
        """
        Count tigers that have at least one valid move.
        
        Args:
            state: The game state
            
        Returns:
            Number of tigers that can move
        """
        try:
            from models.minimax_agent import MinimaxAgent
            return MinimaxAgent.count_movable_tigers(state)
        except (ImportError, AttributeError):
            # Fallback - assume all tigers can move (4)
            return 4
    
    @classmethod
    def _count_threatened_goats(cls, state: GameState) -> int:
        """
        Count goats that can be captured in the next move.
        
        Args:
            state: The game state
            
        Returns:
            Number of goats that can be captured
        """
        try:
            from models.minimax_agent import MinimaxAgent
            return MinimaxAgent.count_threatened_goats(state)
        except (ImportError, AttributeError):
            # Fallback
            return 0
    
    @classmethod
    def _calculate_tiger_dispersion(cls, state: GameState) -> float:
        """
        Calculate the spatial dispersion of tigers (a measure of how spread out they are).
        
        Args:
            state: The game state
            
        Returns:
            Dispersion value between 0.0 and 1.0
        """
        try:
            from models.minimax_agent import MinimaxAgent
            return MinimaxAgent.calculate_tiger_dispersion(state)
        except (ImportError, AttributeError):
            # Fallback - assume medium dispersion
            return 0.5
    
    @classmethod
    def _calculate_goat_edge_preference(cls, state: GameState) -> float:
        """
        Calculate goats' preference for edge positions.
        
        Args:
            state: The game state
            
        Returns:
            Edge preference value between 0.0 and 1.0
        """
        try:
            from models.minimax_agent import MinimaxAgent
            return MinimaxAgent.calculate_goat_edge_preference(state)
        except (ImportError, AttributeError):
            # Fallback - assume medium edge preference
            return 0.5
        
    @classmethod
    def extract_raw_features(cls, state: GameState) -> Dict[str, float]:
        """
        Extract raw features from a game state.
        
        Args:
            state: The game state to extract features from
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Extract basic game state features
        features['goats_placed'] = state.goats_placed
        features['goats_captured'] = state.goats_captured
        
        # Extract derived features by calling minimax agent's functions
        closed_spaces = cls._count_closed_spaces(state)
        features['effective_closed_spaces'] = max(0, closed_spaces - state.goats_captured)
        features['movable_tigers'] = cls._count_movable_tigers(state)
        features['threatened_goats'] = cls._count_threatened_goats(state)
        features['tiger_dispersion'] = cls._calculate_tiger_dispersion(state)
        features['goat_edge_preference'] = cls._calculate_goat_edge_preference(state)
        
        # Turn
        features['turn'] = 1 if state.turn == "TIGER" else 0
        
        return features
    
    @classmethod
    def discretize_features(cls, features: Dict[str, float]) -> Dict[str, int]:
        """
        Discretize continuous features to create keys for the table.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Dictionary with discretized feature values
        """
        discretized = {}
        
        # Direct copy for features that should use exact values
        discretized['goats_placed'] = int(features['goats_placed'])
        discretized['goats_captured'] = int(min(features['goats_captured'], 4))  # Cap at 4
        discretized['turn'] = int(features['turn'])
        
        # Bin effective_closed_spaces: 0, 1, 2+
        value = features['effective_closed_spaces']
        if value >= 2:
            discretized['effective_closed_spaces'] = 2
        else:
            discretized['effective_closed_spaces'] = int(value)
        
        # Bin movable_tigers: 1-2, 3-4
        value = features['movable_tigers']
        if value <= 2:
            discretized['movable_tigers'] = 1
        else:
            discretized['movable_tigers'] = 2
        
        # Bin threatened_goats: 0, 1, 2+
        value = features['threatened_goats']
        if value >= 2:
            discretized['threatened_goats'] = 2
        else:
            discretized['threatened_goats'] = int(value)
        
        # Bin tiger_dispersion: Low, Medium, High
        value = features['tiger_dispersion']
        if value <= 0.4:
            discretized['tiger_dispersion'] = 0  # Low
        elif value <= 0.7:
            discretized['tiger_dispersion'] = 1  # Medium
        else:
            discretized['tiger_dispersion'] = 2  # High
        
        # Bin goat_edge_preference: Low, High
        value = features['goat_edge_preference']
        if value <= 0.4:
            discretized['goat_edge_preference'] = 0  # Low
        else:
            discretized['goat_edge_preference'] = 1  # High
        
        return discretized
    
    @classmethod
    def create_feature_key(cls, features: Dict[str, Any]) -> str:
        """
        Create a feature key string from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature key string
        """
        # Ensure we're using discretized features
        if not all(isinstance(features.get(name, 0), int) for name in cls.FEATURE_NAMES):
            discretized = cls.discretize_features(features)
            return cls.create_key_from_features(discretized)
        
        return cls.create_key_from_features(features)
    
    @classmethod
    def create_key_from_features(cls, features: Dict[str, int]) -> str:
        """
        Create a unique key string from discretized features.
        
        Args:
            features: Dictionary of discretized features
            
        Returns:
            Feature key string
        """
        # Create key as a colon-separated string of feature values in a consistent order
        key_parts = []
        
        for feature_name in cls.FEATURE_NAMES:
            # Use the feature value if available, otherwise use default (0)
            value = features.get(feature_name, 0)
            # Ensure value is an integer
            try:
                value = int(value)
            except (ValueError, TypeError):
                value = 0
                
            key_parts.append(str(value))
        
        return ':'.join(key_parts)
    
    @classmethod
    def features_from_key(cls, feature_key: str) -> Dict[str, int]:
        """
        Parse a feature key back into a feature dictionary.
        
        Args:
            feature_key: Feature key string
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Check if the key is in the old format (contains "=")
        if "=" in feature_key:
            # Return a default dictionary with zeros for all features
            for feature_name in cls.FEATURE_NAMES:
                features[feature_name] = 0
            return features
            
        # Parse feature key (new format: colon-separated values)
        parts = feature_key.split(':')
        
        for i, feature_name in enumerate(cls.FEATURE_NAMES):
            if i < len(parts):
                try:
                    features[feature_name] = int(parts[i])
                except (ValueError, TypeError):
                    # Handle conversion errors by using default value
                    features[feature_name] = 0
            else:
                # Default value for missing features
                features[feature_name] = 0
        
        return features


class WinRateTable:
    """
    Table for storing and retrieving win rate predictions for game states.
    Handles interpolation for unseen states.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the win rate table.
        
        Args:
            storage_path: Directory to save/load the table from
        """
        self.entries = {}  # Dict mapping feature keys to WinRateEntry objects
        self.lock = threading.RLock()  # For thread safety
        
        # Set storage path
        if storage_path is None:
            # Use default path
            self.storage_path = self._get_default_storage_path()
        else:
            self.storage_path = Path(storage_path)
            
        # Create directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
            
        # Load existing table if available
        self._load_table()
        
    def _get_default_storage_path(self) -> Path:
        """Get the default storage path for the table."""
        # Check for environment variable
        env_path = os.environ.get('BAGHCHAL_DATA_DIR')
        if env_path:
            return Path(env_path)
            
        # Default to project's data directory
        # Try to find the project root by checking for backend directory
        current_dir = Path(__file__).parent.parent  # models directory parent (backend)
        data_dir = current_dir / "data"
        
        # Create the data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        return data_dir
    
    def _load_table(self) -> bool:
        """
        Load the win rate table from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        table_path = self.storage_path / "win_rate_table.pkl"
        
        if not table_path.exists():
            return False
            
        try:
            with open(table_path, 'rb') as f:
                self.entries = pickle.load(f)
            return True
        except (IOError, pickle.PickleError) as e:
            print(f"Error loading win rate table: {e}")
            return False
    
    def save_table(self) -> bool:
        """
        Save the win rate table to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        table_path = self.storage_path / "win_rate_table.pkl"
        
        try:
            # Acquire lock to prevent concurrent modifications
            with self.lock:
                # Clean up any entries with old format keys before saving
                cleaned_entries = {}
                for key, entry in self.entries.items():
                    # Skip any key in the old format
                    if "=" in key:
                        continue
                    # Verify the key can be parsed as the right number of components
                    parts = key.split(':')
                    if len(parts) != len(FeatureExtractor.FEATURE_NAMES):
                        continue
                    
                    # Keep only valid entries
                    cleaned_entries[key] = entry
                
                # Replace entries with the cleaned version
                self.entries = cleaned_entries
                
                # Save to disk
                with open(table_path, 'wb') as f:
                    pickle.dump(self.entries, f)
            return True
        except (IOError, pickle.PickleError) as e:
            print(f"Error saving win rate table: {e}")
            return False
    
    def update_from_game_sequence(self, state_sequence: List[GameState], 
                                outcome: float, temporal_weight: bool = True,
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Update the win rate table based on a sequence of game states and their final outcome.
        
        This applies temporal weighting so that states closer to the outcome
        have more influence on the win rate updates.
        
        Args:
            state_sequence: List of game states from a single game
            outcome: Final game outcome (1.0 for Tiger win, 0.0 for Goat win, 0.5 for draw)
            temporal_weight: Whether to apply temporal weighting
            dry_run: If True, don't actually update the table, just return what would change
            
        Returns:
            Dictionary with stats about the update
        """
        # Validation
        if not state_sequence:
            return {
                "states_updated": 0,
                "new_entries": 0,
                "updated_entries": 0
            }
        
        # Track statistics
        updates = {
            "states_updated": 0,
            "new_entries": 0,
            "updated_entries": 0,
            "changes": []
        }
        
        states_count = len(state_sequence)
        
        # Create a list to store entries we'll update
        entries_to_update = []
        
        # First pass - Get current values and prepare changes
        for i, state in enumerate(state_sequence):
            # Extract features from the state
            features = FeatureExtractor.extract_raw_features(state)
            discretized = FeatureExtractor.discretize_features(features)
            feature_key = FeatureExtractor.create_feature_key(discretized)
            
            # Calculate position weight - states closer to the end have more weight
            position_weight = (i + 1) / states_count if temporal_weight else 1.0
            
            # Scale the weight to increase its effect (optional)
            # This makes temporal weighting more pronounced
            position_weight = position_weight ** 1.5  # Raising to a power > 1 increases later weights
            
            # Adjust sample count based on position (more samples for later positions)
            # This ensures states closer to the outcome have more influence
            sample_count = max(1, int(5 * position_weight))
            
            # Try to get the current entry
            entry = self.entries.get(feature_key)
            
            if entry:
                # Entry exists, so we'll update it
                old_win_rate = entry.win_rate
                old_samples = entry.total_samples
                
                # Create a copy to simulate the update for dry run
                new_entry = WinRateEntry()
                new_entry.win_count = entry.win_count
                new_entry.loss_count = entry.loss_count
                new_entry.draw_count = entry.draw_count
                new_entry.total_samples = entry.total_samples
                
                # Update the copy with the outcome
                if outcome == 1.0:  # Tiger win
                    new_entry.win_count += sample_count
                elif outcome == 0.0:  # Goat win
                    new_entry.loss_count += sample_count
                else:  # Draw
                    new_entry.draw_count += sample_count
                
                # Update total samples
                new_entry.total_samples = new_entry.win_count + new_entry.loss_count + new_entry.draw_count
                
                entries_to_update.append({
                    "state_index": i,
                    "feature_key": feature_key,
                    "entry": entry,
                    "is_new": False,
                    "sample_count": sample_count,
                    "old_win_rate": old_win_rate,
                    "new_win_rate": new_entry.win_rate,
                    "old_samples": old_samples,
                    "new_samples": new_entry.total_samples,
                    "position_weight": position_weight
                })
            else:
                # Entry doesn't exist, create a new one
                new_entry = WinRateEntry()
                
                # Update with the outcome
                if outcome == 1.0:  # Tiger win
                    new_entry.win_count += sample_count
                elif outcome == 0.0:  # Goat win
                    new_entry.loss_count += sample_count
                else:  # Draw
                    new_entry.draw_count += sample_count
                
                # Update total samples
                new_entry.total_samples = new_entry.win_count + new_entry.loss_count + new_entry.draw_count
                
                entries_to_update.append({
                    "state_index": i,
                    "feature_key": feature_key,
                    "entry": new_entry,
                    "is_new": True,
                    "sample_count": sample_count,
                    "old_win_rate": 0.5,  # Default prior
                    "new_win_rate": new_entry.win_rate,
                    "old_samples": 0,
                    "new_samples": new_entry.total_samples,
                    "position_weight": position_weight
                })
        
        # Collect changes for reporting
        for update_info in entries_to_update:
            updates["changes"].append({
                "state_index": update_info["state_index"],
                "is_new": update_info["is_new"],
                "old_win_rate": update_info["old_win_rate"],
                "new_win_rate": update_info["new_win_rate"],
                "old_samples": update_info["old_samples"],
                "new_samples": update_info["new_samples"],
                "position_weight": update_info["position_weight"]
            })
        
        # Count statistics
        updates["states_updated"] = len(entries_to_update)
        updates["new_entries"] = sum(1 for update in entries_to_update if update["is_new"])
        updates["updated_entries"] = sum(1 for update in entries_to_update if not update["is_new"])
        
        # If this is just a dry run, don't actually update the table
        if dry_run:
            return updates
        
        # Second pass - Actually update the table
        for update_info in entries_to_update:
            feature_key = update_info["feature_key"]
            sample_count = update_info["sample_count"]
            
            if update_info["is_new"]:
                # Add a new entry
                self.entries[feature_key] = update_info["entry"]
            else:
                # Update existing entry
                entry = self.entries.get(feature_key)
                if entry:
                    if outcome == 1.0:  # Tiger win
                        entry.win_count += sample_count
                    elif outcome == 0.0:  # Goat win
                        entry.loss_count += sample_count
                    else:  # Draw
                        entry.draw_count += sample_count
                    
                    # Update total samples
                    entry.total_samples = entry.win_count + entry.loss_count + entry.draw_count
        
        return updates
    
    def update_from_multi_games(self, 
                           game_sequences: List[Tuple[List[GameState], float]],
                           temporal_weight: bool = True,
                           dry_run: bool = False) -> Dict[str, Any]:
        """
        Update the win rate table from multiple game sequences.
        
        Args:
            game_sequences: List of (state_sequence, outcome) tuples
            temporal_weight: If True, weight states closer to the end more heavily
            dry_run: If True, don't actually update the table, just return what would change
            
        Returns:
            Dictionary with statistics about the update operation
        """
        total_stats = {
            "games_processed": len(game_sequences),
            "total_states_updated": 0,
            "total_new_entries": 0,
            "total_updated_entries": 0,
            "changes_by_game": []
        }
        
        for i, (state_sequence, outcome) in enumerate(game_sequences):
            game_stats = self.update_from_game_sequence(
                state_sequence, outcome, temporal_weight, dry_run
            )
            
            total_stats["total_states_updated"] += game_stats["states_updated"]
            total_stats["total_new_entries"] += game_stats["new_entries"]
            total_stats["total_updated_entries"] += game_stats["updated_entries"]
            
            if dry_run:
                total_stats["changes_by_game"].append({
                    "game_index": i,
                    "changes": game_stats["changes"]
                })
        
        return total_stats
    
    def get_win_rate(self, state: GameState, use_interpolation: bool = True) -> float:
        """
        Get the win rate prediction for a game state.
        
        Args:
            state: The game state to predict for
            use_interpolation: Whether to use interpolation for unknown states
            
        Returns:
            Predicted win rate [0.0, 1.0]
        """
        # Extract features
        raw_features = FeatureExtractor.extract_raw_features(state)
        discretized_features = FeatureExtractor.discretize_features(raw_features)
        key = FeatureExtractor.create_feature_key(discretized_features)
        
        with self.lock:
            # Check if we have a direct entry
            if key in self.entries:
                return self.entries[key].win_rate
                
            # No direct entry, use interpolation if enabled
            if use_interpolation:
                return self._interpolate_win_rate(discretized_features)
                
            # Otherwise use a default win rate
            return self._get_default_win_rate(discretized_features)
    
    def _interpolate_win_rate(self, query_features: Dict[str, int], k: int = 3) -> float:
        """
        Interpolate win rate based on similar feature combinations.
        
        Args:
            query_features: Features to interpolate for
            k: Number of nearest neighbors to consider
            
        Returns:
            Interpolated win rate
        """
        with self.lock:
            if not self.entries:
                return 0.5  # No data, return prior
            
            # Calculate distances to all entries
            entries_with_distance = []
            
            for key, entry in self.entries.items():
                # Parse features from the key
                table_features = FeatureExtractor.features_from_key(key)
                
                # Calculate distance between feature sets
                distance = self._calculate_feature_distance(query_features, table_features)
                
                # Store entry with its distance
                entries_with_distance.append((distance, entry))
            
            # Sort by distance (closest first)
            entries_with_distance.sort(key=lambda x: x[0])
            
            # Take top k entries
            top_k = entries_with_distance[:k]
            
            if not top_k:
                return 0.5  # No neighbors, return prior
            
            # Weight by inverse distance and confidence
            total_weight = 0
            weighted_sum = 0
            
            for distance, entry in top_k:
                # Convert distance to weight (closer = higher weight)
                # Add small epsilon to avoid division by zero
                if distance < 0.001:
                    # Exact match, no need for interpolation
                    return entry.win_rate
                    
                # Weight is inverse of distance, multiplied by confidence
                weight = (1.0 / distance) * entry.confidence
                
                weighted_sum += entry.win_rate * weight
                total_weight += weight
            
            # Calculate weighted average
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return 0.5  # Fallback to prior
    
    def _calculate_feature_distance(self, features1: Dict[str, int], features2: Dict[str, int]) -> float:
        """
        Calculate the distance between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Distance measure between the features
        """
        # Define base weights for different features
        base_weights = {
            'turn': 1.5,                     # Turn is significant
            'goats_placed': 1.0,             # Basic game state
            'effective_closed_spaces': 1.2,  # Strategic metric
            'movable_tigers': 1.0,           # Tactical consideration
            'threatened_goats': 1.0,         # Tactical consideration
            'tiger_dispersion': 0.8,         # Spatial factor
            'goat_edge_preference': 0.8      # Spatial factor
        }
        
        # Adjust goats_captured weight based on context
        goats_captured_weight = 1.5  # Base weight (default importance)
        
        # Critical threshold - capture count becomes extremely important
        if features1['goats_captured'] == 4 or features2['goats_captured'] == 4:
            goats_captured_weight = 3.0  # Much more important
        # Similar capture counts - less distinguishing
        elif abs(features1['goats_captured'] - features2['goats_captured']) <= 1:
            goats_captured_weight = 1.0  # Lower weight
            
        # Combine base weights with dynamic goats_captured weight
        weights = {**base_weights, 'goats_captured': goats_captured_weight}
        
        # Calculate distances for each feature
        distances = {}
        
        # Categorical features (exact match or not)
        for feature in ['turn']:
            if features1[feature] != features2[feature]:
                distances[feature] = 1.0
            else:
                distances[feature] = 0.0
                
        # Ordinal features with normalized distances
        distances['goats_placed'] = abs(features1['goats_placed'] - features2['goats_placed']) / 20.0
        distances['goats_captured'] = abs(features1['goats_captured'] - features2['goats_captured']) / 5.0
        
        # Binned features - distance depends on the number of bins
        distances['effective_closed_spaces'] = abs(features1['effective_closed_spaces'] - features2['effective_closed_spaces']) / 2.0
        distances['movable_tigers'] = abs(features1['movable_tigers'] - features2['movable_tigers']) / 1.0  # only 2 bins
        distances['threatened_goats'] = abs(features1['threatened_goats'] - features2['threatened_goats']) / 2.0
        distances['tiger_dispersion'] = abs(features1['tiger_dispersion'] - features2['tiger_dispersion']) / 2.0
        distances['goat_edge_preference'] = abs(features1['goat_edge_preference'] - features2['goat_edge_preference']) / 1.0  # only 2 bins
        
        # Calculate weighted sum of distances
        total_distance = sum(weights[feature] * distances[feature] for feature in weights)
        
        # Normalize by sum of weights
        return total_distance / sum(weights.values())
    
    def _get_default_win_rate(self, features: Dict[str, int]) -> float:
        """
        Get a default win rate for features with no data.
        
        Args:
            features: Discretized feature dictionary
            
        Returns:
            Default win rate prediction
        """
        # Critical threshold - tiger is one capture away from winning
        if features['goats_captured'] == 4:
            # Tigers are heavily favored but not guaranteed to win
            # Factor in the number of threatened goats
            if features['threatened_goats'] > 0:
                return 0.85  # Very high tiger win probability with threatened goats
            return 0.75  # Still high tiger win probability
        
        # High capture count favors tigers significantly
        elif features['goats_captured'] == 3:
            return 0.65
            
        # Mid-game with 2 captures - slight tiger advantage
        elif features['goats_captured'] == 2:
            return 0.55
            
        # Early captures - balanced situation
        elif features['goats_captured'] == 1:
            return 0.5
            
        # Late placement phase with no captures favors goats
        elif features['goats_placed'] >= 15 and features['goats_captured'] == 0:
            return 0.4  # Goat advantage
            
        # Movement phase with no captures significantly favors goats
        elif features['goats_placed'] == 20 and features['goats_captured'] == 0:
            if features['effective_closed_spaces'] > 0:
                return 0.3  # Stronger goat advantage with closed spaces
            return 0.35  # Goat advantage
            
        # Otherwise balanced
        return 0.5
    
    def get_table_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the table.
        
        Returns:
            Dictionary with table statistics
        """
        entries = list(self.entries.values())
        
        if not entries:
            return {
                "total_entries": 0,
                "total_samples": 0,
                "avg_win_rate": 0.0,
                "avg_confidence": 0.0
            }
        
        total_samples = sum(entry.total_samples for entry in entries)
        
        # Calculate average win rate weighted by sample count
        if total_samples > 0:
            avg_win_rate = sum(entry.win_rate * entry.total_samples for entry in entries) / total_samples
        else:
            avg_win_rate = 0.0
            
        # Calculate average confidence
        avg_confidence = sum(entry.confidence for entry in entries) / len(entries)
        
        return {
            "total_entries": len(entries),
            "total_samples": total_samples,
            "avg_win_rate": avg_win_rate,
            "avg_confidence": avg_confidence
        }

    def _parse_features_from_key(self, feature_key: str) -> Dict[str, int]:
        """
        Parse features from a feature key.
        
        Args:
            feature_key: Feature key string
            
        Returns:
            Dictionary of features
        """
        # Use the standard method from FeatureExtractor with error handling
        return FeatureExtractor.features_from_key(feature_key)


class WinRatePredictor:
    """
    Main interface for win rate prediction.
    Uses a WinRateTable internally.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the win rate predictor.
        
        Args:
            storage_path: Directory to save/load the table from
        """
        self.table = WinRateTable(storage_path)
    
    def predict_win_rate(self, state: GameState) -> float:
        """
        Predict the win rate for a game state.
        
        Args:
            state: The game state to predict for
            
        Returns:
            Predicted win rate [0.0, 1.0]
        """
        return self.table.get_win_rate(state)
    
    def update_from_game_sequence(self, state_sequence: List[GameState], 
                                outcome: float, temporal_weight: bool = True,
                                dry_run: bool = False) -> Dict[str, Any]:
        """
        Update the win rate table based on a sequence of game states and their final outcome.
        
        This applies temporal weighting so that states closer to the outcome
        have more influence on the win rate updates.
        
        Args:
            state_sequence: List of game states from a single game
            outcome: Final game outcome (1.0 for Tiger win, 0.0 for Goat win, 0.5 for draw)
            temporal_weight: Whether to apply temporal weighting
            dry_run: If True, don't actually update the table, just return what would change
            
        Returns:
            Dictionary with stats about the update
        """
        return self.table.update_from_game_sequence(state_sequence, outcome, temporal_weight, dry_run)
    
    def update_from_multi_games(self, 
                           game_sequences: List[Tuple[List[GameState], float]],
                           temporal_weight: bool = True,
                           dry_run: bool = False) -> Dict[str, Any]:
        """
        Update the predictor from multiple game sequences.
        
        Args:
            game_sequences: List of (state_sequence, outcome) tuples
            temporal_weight: If True, weight states closer to the end more heavily
            dry_run: If True, don't actually update the table, just report changes
            
        Returns:
            Dictionary with update statistics
        """
        return self.table.update_from_multi_games(
            game_sequences, temporal_weight, dry_run
        )
    
    def save(self) -> bool:
        """
        Save the win rate table to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        return self.table.save_table()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the win rate table.
        
        Returns:
            Dictionary with table statistics
        """
        entries = list(self.table.entries.values())
        
        if not entries:
            return {
                "total_entries": 0,
                "total_samples": 0,
                "avg_win_rate": 0.0,
                "avg_confidence": 0.0,
                "feature_coverage": {}
            }
        
        # Count the total samples directly from the entries
        total_samples = sum(entry.total_samples for entry in entries)
        
        # Calculate average win rate weighted by sample count
        if total_samples > 0:
            avg_win_rate = sum(entry.win_rate * entry.total_samples for entry in entries) / total_samples
        else:
            avg_win_rate = 0.0
            
        # Calculate average confidence
        avg_confidence = sum(entry.confidence for entry in entries) / len(entries)
        
        # Calculate feature distribution statistics
        feature_coverage = {}
        
        # For each feature, count how many entries exist for each feature value
        for feature_name in FeatureExtractor.FEATURE_NAMES:
            feature_coverage[feature_name] = {}
            
            # Extract the range of possible values for this feature
            if feature_name in FeatureExtractor.FEATURE_RANGES:
                feature_range = FeatureExtractor.FEATURE_RANGES[feature_name]
                
                # Initialize counters for each possible value
                for value in feature_range:
                    feature_coverage[feature_name][str(value)] = 0
                
                # Count entries for each feature value
                for feature_key in self.table.entries:
                    try:
                        # Parse the feature key
                        features = self.table._parse_features_from_key(feature_key)
                        if feature_name in features:
                            value = features[feature_name]
                            value_str = str(value)
                            if value_str in feature_coverage[feature_name]:
                                feature_coverage[feature_name][value_str] += 1
                    except Exception:
                        # Skip any problematic keys
                        continue
        
        # Get regions needing more data (low sample count)
        sparse_regions = []
        if entries:
            try:
                entries_by_samples = sorted(
                    [(key, self.table.entries[key].total_samples) for key in self.table.entries],
                    key=lambda x: x[1]
                )[:5]  # Get 5 regions with fewest samples
                
                for key, samples in entries_by_samples:
                    if samples < 10:  # Threshold for "needing more data"
                        try:
                            features = self.table._parse_features_from_key(key)
                            sparse_regions.append({
                                "features": features,
                                "samples": samples
                            })
                        except Exception:
                            # Skip any problematic keys
                            continue
            except Exception:
                # If there's any error in processing, just return an empty list
                sparse_regions = []
        
        return {
            "total_entries": len(entries),
            "total_samples": total_samples,
            "avg_win_rate": avg_win_rate,
            "avg_confidence": avg_confidence,
            "feature_coverage": feature_coverage,
            "sparse_regions": sparse_regions
        }

    def get_features(self, state: GameState) -> Dict[str, int]:
        """
        Extract and discretize features from a game state.
        
        Args:
            state: Game state to extract features from
            
        Returns:
            Dictionary with discretized features
        """
        raw_features = FeatureExtractor.extract_raw_features(state)
        return FeatureExtractor.discretize_features(raw_features)
        
    def get_entry(self, feature_key: str) -> Optional[WinRateEntry]:
        """
        Get the entry for a specific feature key.
        
        Args:
            feature_key: Feature key to look up
            
        Returns:
            WinRateEntry if found, None otherwise
        """
        return self.table.entries.get(feature_key)
            
    def set_entry(self, feature_key: str, entry: WinRateEntry) -> None:
        """
        Set the entry for a specific feature key.
        
        Args:
            feature_key: Feature key to set
            entry: WinRateEntry to store
        """
        self.table.entries[feature_key] = entry
    
    def get_win_rate(self, state: GameState) -> float:
        """
        Get the win rate for a game state, using interpolation if needed.
        
        Args:
            state: Game state to look up
            
        Returns:
            Interpolated win rate
        """
        return self.table.get_win_rate(state)


# Singleton instance
_win_rate_predictor_instance = None

def get_win_rate_predictor(storage_path: str = None) -> WinRatePredictor:
    """
    Get the win rate predictor singleton instance.
    
    Args:
        storage_path: Directory to save/load the table from
        
    Returns:
        WinRatePredictor instance
    """
    global _win_rate_predictor_instance
    
    if _win_rate_predictor_instance is None:
        _win_rate_predictor_instance = WinRatePredictor(storage_path)
        
    return _win_rate_predictor_instance

def reset_win_rate_predictor() -> None:
    """Reset the win rate predictor singleton (for testing)."""
    global _win_rate_predictor_instance
    _win_rate_predictor_instance = None 