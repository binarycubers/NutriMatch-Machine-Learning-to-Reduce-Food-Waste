import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class DataPreprocessor:
    def __init__(self, base_dir=None):
        """
        Initialize the DataPreprocessor class
        base_dir: Base directory for all data operations (should be your project root)
        """
        # Set project root directory
        self.base_dir = base_dir if base_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Set up log file
        self.log_file = os.path.join(self.base_dir, 'logs',
                                   f'preprocessing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.join(self.base_dir, 'logs'), exist_ok=True)
        
        # Directory structure
        self.data_dirs = {
            'raw': os.path.join(self.base_dir, 'data', 'raw'),
            'processed': os.path.join(self.base_dir, 'data', 'processed'),
            'interim': os.path.join(self.base_dir, 'data', 'interim'),
            'external': os.path.join(self.base_dir, 'data', 'external'),
            'logs': os.path.join(self.base_dir, 'logs'),
            'models': os.path.join(self.base_dir, 'models')
        }
        
        self.create_directories()

        self.df = None
        self.scaler = MinMaxScaler()
        self.original_shape = None
        self.normalized_columns = []
        self.missing_values_handled = False

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_path in self.data_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            self.log_message(f"Directory exists: {dir_path}")

    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(log_message)

    def load_data(self, file_name, source_dir=None):
        """
        Load the dataset from specified directory
        file_name: Name of the data file
        source_dir: Custom source directory (optional)
        """
        try:
            # Use custom directory if provided, else use default raw data directory
            source_path = os.path.join(source_dir if source_dir else self.data_dirs['raw'], file_name)
            self.log_message(f"Attempting to load data from: {source_path}")
            
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"File not found at: {source_path}")
                
            self.df = pd.read_csv(source_path)
            self.original_shape = self.df.shape
            self.log_message(f"Data loaded successfully from {source_path}. Shape: {self.df.shape}")

            # Save backup
            backup_path = os.path.join(self.data_dirs['interim'], f'backup_{datetime.now().strftime("%Y%m%d")}_{file_name}')
            self.df.to_csv(backup_path, index=False)
            self.log_message(f"Backup created at {backup_path}")

            return True
        except Exception as e:
            self.log_message(f"Error loading data: {str(e)}")
            return False

    def save_processed_data(self, file_name, custom_dir=None):
        """Save the processed dataset"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name_with_timestamp = f"{os.path.splitext(file_name)[0]}_{timestamp}{os.path.splitext(file_name)[1]}"
            save_path = os.path.join(custom_dir if custom_dir else self.data_dirs['processed'],
                                   file_name_with_timestamp)
            self.df.to_csv(save_path, index=False)
            self.log_message(f"Processed data saved to {save_path}")
            return save_path
        except Exception as e:
            self.log_message(f"Error saving data: {e}")
            return None

def main():
    # Initialize preprocessor - no need for custom base_dir if running from project root
    preprocessor = DataPreprocessor()
    
    # Print directory structure for debugging
    print("\nCurrent directory structure:")
    for dir_type, dir_path in preprocessor.data_dirs.items():
        print(f"{dir_type.upper():<10}: {dir_path}")

    # Load data - no need for source_dir since we're using the proper raw directory
    data_loaded = preprocessor.load_data('Item_FullList.csv')
    
    if not data_loaded:
        print("\nFailed to load data. Possible solutions:")
        print(f"1. Ensure 'Item_FullList.csv' exists in: {preprocessor.data_dirs['raw']}")
        print("2. Check the file permissions")
        print("3. Verify the file is not open in another program")
        return

    # Add your preprocessing steps here
    # preprocessor.df = preprocessor.df.dropna()  # Example
    
    # Save processed data
    processed_path = preprocessor.save_processed_data('processed_data.csv')
    if processed_path:
        print(f"\nSuccessfully processed data saved to: {processed_path}")

if __name__ == "__main__":
    main()