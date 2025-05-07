import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import joblib

class DataSplitter:
    def __init__(self, base_dir=None):
        """
        Initialize DataSplitter with logging functionality
        base_dir: Base directory for all data operations
        """
        # Set project root directory
        self.base_dir = base_dir if base_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Set up log file
        self.log_file = os.path.join(self.base_dir, 'logs',
                                   f'data_splitting_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        # Directory structure
        self.data_dirs = {
            'engineered': os.path.join(self.base_dir, 'data', 'engineered'),
            'split': os.path.join(self.base_dir, 'data', 'split'),
            'logs': os.path.join(self.base_dir, 'logs')
        }

        # Create necessary directories
        for dir_path in self.data_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            self.log_message(f"Directory exists: {dir_path}")

        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(log_message)

    def load_data(self, input_file):
        """Load the engineered features dataset"""
        try:
            input_path = os.path.join(self.data_dirs['engineered'], input_file)
            self.log_message(f"Loading data from: {input_path}")

            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")

            self.data = pd.read_csv(input_path)
            self.log_message(f"Data loaded successfully. Shape: {self.data.shape}")
            self.log_message(f"Available columns: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            self.log_message(f"Error loading data: {str(e)}")
            return False

    def prepare_data(self, target_column='Protein',
                    features_to_exclude=['Date', 'Item Description', 'Item Code',
                                       'Unit Price', 'Total Price']):
        """
        Prepare data by separating features and target
        """
        try:
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Create target variable
            self.y = self.data[target_column]

            # Create feature set
            exclude_cols = [col for col in features_to_exclude if col in self.data.columns]
            exclude_cols.append(target_column)
            self.X = self.data.drop(columns=exclude_cols)

            self.log_message(f"Data prepared successfully:")
            self.log_message(f"Features shape: {self.X.shape}")
            self.log_message(f"Target shape: {self.y.shape}")
            self.log_message(f"Features used: {self.X.columns.tolist()}")
            return True
        except Exception as e:
            self.log_message(f"Error preparing data: {str(e)}")
            return False

    def perform_train_test_split(self, test_size=0.2, random_state=42):
        """Perform basic train-test split"""
        try:
            if self.X is None or self.y is None:
                raise ValueError("Data not prepared. Run prepare_data first.")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, shuffle=True
            )

            self.log_message("Train-Test Split Results:")
            self.log_message(f"X_train shape: {self.X_train.shape}")
            self.log_message(f"X_test shape: {self.X_test.shape}")
            self.log_message(f"y_train shape: {self.y_train.shape}")
            self.log_message(f"y_test shape: {self.y_test.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error in train-test split: {str(e)}")
            return False

    def create_validation_set(self, val_size=0.2):
        """Create a validation set from training data"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Train-test split not performed yet.")

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=val_size, random_state=42
            )

            self.log_message("Validation Split Results:")
            self.log_message(f"X_train shape: {self.X_train.shape}")
            self.log_message(f"X_val shape: {self.X_val.shape}")
            self.log_message(f"y_train shape: {self.y_train.shape}")
            self.log_message(f"y_val shape: {self.y_val.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error creating validation set: {str(e)}")
            return False

    def save_split_data(self):
        """Save the split datasets"""
        try:
            if self.X_train is None:
                raise ValueError("No data to save. Perform splits first.")

            # Save splits
            splits = {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'X_val': self.X_val,
                'y_val': self.y_val
            }

            # Save each split as CSV
            for name, data in splits.items():
                filepath = os.path.join(self.data_dirs['split'], f"{name}.csv")
                data.to_csv(filepath, index=False)
                self.log_message(f"Saved {name} to {filepath}")

            # Save summary statistics
            stats_path = os.path.join(self.data_dirs['split'], f"split_summary_{self.timestamp}.txt")
            with open(stats_path, 'w') as f:
                f.write("Data Splitting Summary\n")
                f.write("=====================\n\n")
                f.write(f"Original data shape: {self.data.shape}\n")
                f.write(f"Training set shape: {self.X_train.shape}\n")
                f.write(f"Validation set shape: {self.X_val.shape}\n")
                f.write(f"Test set shape: {self.X_test.shape}\n")

            self.log_message(f"Split summary saved to: {stats_path}")
            return True
        except Exception as e:
            self.log_message(f"Error saving split data: {str(e)}")
            return False

def main():
    # Initialize splitter with project root directory
    base_dir = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste"  # Change this to your project directory
    splitter = DataSplitter(base_dir)

    # Load the engineered features file
    input_file = "engineered_features.csv"  # Change this to your engineered features file name
    if not splitter.load_data(input_file):
        print("\nFailed to load data. Please check the logs for details.")
        return

    # Prepare data
    if not splitter.prepare_data(
        target_column='Protein',  # Change this to your target column
        features_to_exclude=['Date', 'Item Description', 'Item Code', 'Unit Price', 'Total Price']
    ):
        print("\nFailed to prepare data. Please check the logs for details.")
        return

    # Perform train-test split
    if not splitter.perform_train_test_split(test_size=0.2):
        print("\nFailed to perform train-test split. Please check the logs for details.")
        return

    # Create validation set
    if not splitter.create_validation_set(val_size=0.2):
        print("\nFailed to create validation set. Please check the logs for details.")
        return

    # Save split data
    if splitter.save_split_data():
        print("\nData splitting completed successfully!")
    else:
        print("\nFailed to save split data. Please check the logs for details.")

if __name__ == "__main__":
    main()