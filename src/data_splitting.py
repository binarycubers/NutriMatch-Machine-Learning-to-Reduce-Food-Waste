import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import joblib

class DataSplitter:
    def __init__(self, input_path):
        """
        Initialize DataSplitter
        input_path: Path to your engineered features CSV
        """
        self.data = pd.read_csv(input_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Print the available columns
        print("Available columns:", self.data.columns.tolist())

    def prepare_data(self, target_column='Protein',
                    features_to_exclude=['Date', 'Item Description', 'Item Code',
                                       'Unit Price', 'Total Price']):
        """
        Prepare data by separating features and target
        target_column: Column to predict
        features_to_exclude: Columns to exclude from features
        """
        try:
            # Make sure target column exists
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            # Create target variable
            self.y = self.data[target_column]

            # Create feature set by excluding specified columns and target
            features_to_exclude.append(target_column)
            self.X = self.data.drop(columns=features_to_exclude)

            print(f"\nData prepared successfully:")
            print(f"Features shape: {self.X.shape}")
            print(f"Target shape: {self.y.shape}")
            print("\nFeatures used:", self.X.columns.tolist())
            return True
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False

    def perform_train_test_split(self, test_size=0.2, random_state=42):
        """
        Perform basic train-test split
        """
        try:
            if self.X is None or self.y is None:
                raise ValueError("Data not prepared. Run prepare_data first.")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True
            )

            print("\nTrain-Test Split Results:")
            print(f"X_train shape: {self.X_train.shape}")
            print(f"X_test shape: {self.X_test.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"y_test shape: {self.y_test.shape}")

            return True
        except Exception as e:
            print(f"Error in train-test split: {e}")
            return False

    def create_validation_set(self, val_size=0.2):
        """
        Create a validation set from training data
        """
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Train-test split not performed yet.")

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train,
                self.y_train,
                test_size=val_size,
                random_state=42
            )

            print("\nValidation Split Results:")
            print(f"X_train shape: {self.X_train.shape}")
            print(f"X_val shape: {self.X_val.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"y_val shape: {self.y_val.shape}")

            return True
        except Exception as e:
            print(f"Error creating validation set: {e}")
            return False

    def save_split_data(self, output_dir):
        """
        Save the split datasets
        """
        try:
            if self.X_train is None:
                raise ValueError("No data to save. Perform splits first.")

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save splits
            splits = {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test
            }

            # Add validation set if it exists
            if hasattr(self, 'X_val'):
                splits.update({
                    'X_val': self.X_val,
                    'y_val': self.y_val
                })

            # Save each split as CSV
            for name, data in splits.items():
                filepath = os.path.join(output_dir, f"{name}_{self.timestamp}.csv")
                data.to_csv(filepath, index=False)
                print(f"Saved {name} to {filepath}")

            return True
        except Exception as e:
            print(f"Error saving split data: {e}")
            return False

def main():
    # Set your paths
    input_path = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste\data\engineered\engineered_features_20250506_192430.csv"
    output_dir = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste\data\split"

    # Initialize DataSplitter
    splitter = DataSplitter(input_path)

    # Prepare data (specify your target column)
    splitter.prepare_data(
        target_column='Protein',  # Change this to your target column
        features_to_exclude=[
            'Date',
            'Item Description',
            'Item Code',
            'Unit Price',
            'Total Price'
        ]
    )

    # Perform train-test split
    splitter.perform_train_test_split(test_size=0.2)

    # Create validation set
    splitter.create_validation_set(val_size=0.2)

    # Save split data
    splitter.save_split_data(output_dir)

if __name__ == "__main__":
    main()