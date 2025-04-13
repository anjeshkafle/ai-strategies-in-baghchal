"""
Google Sheets synchronization for simulation results.

This module provides a simple way to sync simulation results to a Google Sheet
via a Google Apps Script web app.
"""

import requests
import time
import json
import os
from typing import List, Dict, Any

class GoogleSheetsSync:
    """
    Simple class to sync data to Google Sheets via a web app URL.
    """
    
    def __init__(self, webapp_url=None, batch_size=100, output_dir="simulation_results"):
        """
        Initialize the sync helper.
        
        Args:
            webapp_url: URL of the Google Apps Script web app
            batch_size: Number of rows to batch before syncing
            output_dir: Directory where simulation results are stored
        """
        self.webapp_url = webapp_url
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.buffer_file = os.path.join(output_dir, "sheets_sync_buffer.json")
        self.row_buffer = []
        self.enabled = bool(webapp_url)  # Enable only if URL is provided
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load any existing buffer from file
        if self.enabled:
            self._load_buffer()
            print(f"Google Sheets sync enabled (batch size: {batch_size}, {len(self.row_buffer)} pending rows)")
        else:
            print("Google Sheets sync disabled (no URL provided)")
    
    def _load_buffer(self):
        """Load the row buffer from the persistent file if it exists."""
        if os.path.exists(self.buffer_file):
            try:
                with open(self.buffer_file, 'r') as f:
                    self.row_buffer = json.load(f)
                print(f"Loaded {len(self.row_buffer)} pending rows from {self.buffer_file}")
            except Exception as e:
                print(f"Error loading sync buffer file: {e}")
                self.row_buffer = []
    
    def _save_buffer(self):
        """Save the current row buffer to the persistent file."""
        if self.enabled and self.row_buffer:
            try:
                with open(self.buffer_file, 'w') as f:
                    json.dump(self.row_buffer, f)
            except Exception as e:
                print(f"Error saving sync buffer file: {e}")
    
    def _clear_buffer_file(self):
        """Delete the buffer file after successful sync."""
        if os.path.exists(self.buffer_file):
            try:
                os.remove(self.buffer_file)
            except Exception as e:
                print(f"Error removing sync buffer file: {e}")
    
    def add_row(self, row_dict: Dict[str, Any], headers: List[str]):
        """
        Add a row to the buffer and sync if batch size is reached.
        
        Args:
            row_dict: Dictionary containing row data
            headers: List of column headers for ordering values
            
        Returns:
            True if sync was triggered, False otherwise
        """
        if not self.enabled:
            return False
        
        # Convert dict to list in the correct order
        row_values = [row_dict.get(header, "") for header in headers]
        self.row_buffer.append(row_values)
        
        # Update the buffer file immediately
        self._save_buffer()
        
        # Sync if batch size is reached
        if len(self.row_buffer) >= self.batch_size:
            return self.sync()
            
        return False
    
    def sync(self, force=False):
        """
        Sync buffered rows to Google Sheets.
        
        Args:
            force: Whether to sync even if buffer is smaller than batch size
            
        Returns:
            True if sync was successful, False otherwise
        """
        if not self.enabled:
            return False
            
        if not self.row_buffer:
            return False
            
        if not force and len(self.row_buffer) < self.batch_size:
            return False
        
        try:
            # Send data to Google Sheets web app
            response = requests.post(
                self.webapp_url,
                json=self.row_buffer,
                headers={"Content-Type": "application/json"},
                timeout=10  # Add timeout to prevent hanging
            )
            
            # Check if sync was successful
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    print(f"Successfully synced {len(self.row_buffer)} rows to Google Sheets")
                    self.row_buffer.clear()  # Clear the buffer after successful sync
                    self._clear_buffer_file()  # Delete the buffer file
                    return True
                else:
                    print(f"Error syncing to Google Sheets: {result.get('message', 'Unknown error')}")
            else:
                print(f"Error syncing to Google Sheets: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Exception during Google Sheets sync: {str(e)}")
            
        # If we got here, sync failed but don't clear buffer - we'll retry later
        # Make sure buffer file is up to date
        self._save_buffer()
        return False 