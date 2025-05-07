import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

class FeatureEngineer:
    def __init__(self, base_dir=None):
        """
        Initialize the FeatureEngineer class
        base_dir: Base directory for all data operations (should be your project root)
        """
        # Set project root directory
        self.base_dir = base_dir if base_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Set up log file
        self.log_file = os.path.join(self.base_dir, 'logs',
                                   f'feature_engineering_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        # Directory structure
        self.data_dirs = {
            'processed': os.path.join(self.base_dir, 'data', 'processed'),
            'engineered': os.path.join(self.base_dir, 'data', 'engineered'),
            'logs': os.path.join(self.base_dir, 'logs')
        }

        # Create necessary directories
        for dir_path in self.data_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            self.log_message(f"Directory exists: {dir_path}")

        self.df = None
        self.scaler = StandardScaler()
        self.original_columns = None

    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(log_message)

    def load_weekly_data(self, file_name='weekly_food_waste_20250507_000105.csv'):
        """Load the weekly dataset from a fixed file name"""
        try:
            source_path = os.path.join(self.data_dirs['processed'], file_name)
            self.log_message(f"Loading weekly data from: {source_path}")
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"File not found at: {source_path}")
            self.df = pd.read_csv(source_path)
            self.original_columns = self.df.columns.tolist()
            self.log_message(f"Loaded weekly data successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error loading weekly data: {str(e)}")
            return False

    def create_nutrient_ratios(self):
        """Create new features based on nutrient ratios"""
        try:
            # Protein to Fat ratio
            if 'Protein' in self.df.columns and 'Fat' in self.df.columns:
                self.df['Protein_Fat_Ratio'] = self.df['Protein'] / (self.df['Fat'] + 1e-6)

            # Carbs to Protein ratio
            if 'Carbohydrates' in self.df.columns and 'Protein' in self.df.columns:
                self.df['Carbs_Protein_Ratio'] = self.df['Carbohydrates'] / (self.df['Protein'] + 1e-6)

            # Fiber to Carbs ratio
            if 'Fiber' in self.df.columns and 'Carbohydrates' in self.df.columns:
                self.df['Fiber_Carbs_Ratio'] = self.df['Fiber'] / (self.df['Carbohydrates'] + 1e-6)

            self.log_message("Created nutrient ratio features successfully")
            return True
        except Exception as e:
            self.log_message(f"Error creating nutrient ratios: {e}")
            return False

    def create_nutrient_interactions(self):
        """Create interaction features between nutrients"""
        try:
            # Protein x Fat interaction
            if 'Protein' in self.df.columns and 'Fat' in self.df.columns:
                self.df['Protein_Fat_Interaction'] = self.df['Protein'] * self.df['Fat']

            # Protein x Carbs interaction
            if 'Protein' in self.df.columns and 'Carbohydrates' in self.df.columns:
                self.df['Protein_Carbs_Interaction'] = self.df['Protein'] * self.df['Carbohydrates']

            # Fat x Carbs interaction
            if 'Fat' in self.df.columns and 'Carbohydrates' in self.df.columns:
                self.df['Fat_Carbs_Interaction'] = self.df['Fat'] * self.df['Carbohydrates']

            self.log_message("Created nutrient interaction features successfully")
            return True
        except Exception as e:
            self.log_message(f"Error creating nutrient interactions: {e}")
            return False

    def create_time_features(self):
        """Create time-based features"""
        try:
            if 'Week' in self.df.columns:
                # Create cyclical features for week number
                self.df['Week_Sin'] = np.sin(2 * np.pi * self.df['Week']/52.0)
                self.df['Week_Cos'] = np.cos(2 * np.pi * self.df['Week']/52.0)

            self.log_message("Created time-based features successfully")
            return True
        except Exception as e:
            self.log_message(f"Error creating time features: {e}")
            return False

    def normalize_features(self):
        """Normalize numerical features"""
        try:
            # Select numerical columns excluding date and week-related columns
            exclude_cols = ['Year', 'Week', 'Week_Start', 'Week_End']
            num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            cols_to_normalize = [col for col in num_cols if col not in exclude_cols]

            self.df[cols_to_normalize] = self.scaler.fit_transform(self.df[cols_to_normalize])
            self.log_message(f"Normalized {len(cols_to_normalize)} features successfully")
            return True
        except Exception as e:
            self.log_message(f"Error normalizing features: {e}")
            return False

    def save_engineered_features(self, file_name='engineered_features.csv'):
        """Save the engineered features to a fixed file name"""
        try:
            output_path = os.path.join(self.data_dirs['engineered'], file_name)
            self.df.to_csv(output_path, index=False)
            self.log_message(f"Saved engineered features to: {output_path}")

            # Save feature engineering summary
            summary_file = os.path.join(self.data_dirs['engineered'], 'feature_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Feature Engineering Summary\n")
                f.write("=========================\n\n")
                f.write(f"Original columns: {len(self.original_columns)}\n")
                f.write(f"Engineered columns: {len(self.df.columns)}\n")
                f.write(f"New features created: {len(self.df.columns) - len(self.original_columns)}\n\n")
                f.write("New features:\n")
                for col in self.df.columns:
                    if col not in self.original_columns:
                        f.write(f"- {col}\n")

            return output_path
        except Exception as e:
            self.log_message(f"Error saving engineered features: {e}")
            return None

def main():
    try:
        # Set your project root directory here
        base_dir = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste"  # Change this to your project directory
        fe = FeatureEngineer(base_dir)

        # Load weekly data
        if not fe.load_weekly_data('weekly_food_waste_20250507_000105.csv'):
            print("Failed to load weekly data. Check the logs for details.")
            return

        # Create features
        fe.create_nutrient_ratios()
        fe.create_nutrient_interactions()
        fe.create_time_features()
        fe.normalize_features()

        # Save engineered features
        output_path = fe.save_engineered_features('engineered_features.csv')

        if output_path:
            print(f"\nFeature engineering completed successfully!")
            print(f"Output file: {output_path}")
        else:
            print("\nFeature engineering completed with errors.")

    except Exception as e:
        print(f"\nCritical error during feature engineering: {e}")

if __name__ == "__main__":
    main()