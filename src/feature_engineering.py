import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from datetime import datetime
from glob import glob

class FeatureEngineer:
    def __init__(self, input_data_path=None):
        """
        Initialize the FeatureEngineer class
        input_data_path: Path to the preprocessed data (optional)
        """
        if input_data_path is None:
            # Find the most recent preprocessed file automatically
            processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'data', 'processed')
            files = glob(os.path.join(processed_dir, 'processed_data_*.csv'))
            if not files:
                raise FileNotFoundError(f"No processed data files found in {processed_dir}")
            # Get the most recent file
            input_data_path = max(files, key=os.path.getctime)
        
        self.input_path = input_data_path
        self.df = pd.read_csv(input_data_path)
        self.scaler = StandardScaler()
        self.original_columns = self.df.columns.tolist()
        print(f"Loaded data from: {input_data_path}")

    def create_nutrient_ratios(self):
        """Create new features based on nutrient ratios"""
        try:
            # Protein to Calorie ratio
            if 'Protein' in self.df.columns and 'Calories' in self.df.columns:
                self.df['Protein_Calorie_Ratio'] = self.df['Protein'] / self.df['Calories']

            # Fat to Calorie ratio
            if 'Fat' in self.df.columns and 'Calories' in self.df.columns:
                self.df['Fat_Calorie_Ratio'] = self.df['Fat'] / self.df['Calories']

            # Carbs to Calorie ratio
            if 'Carbohydrates' in self.df.columns and 'Calories' in self.df.columns:
                self.df['Carbs_Calorie_Ratio'] = self.df['Carbohydrates'] / self.df['Calories']

            # Fiber to Carbs ratio
            if 'Fiber' in self.df.columns and 'Carbohydrates' in self.df.columns:
                self.df['Fiber_Carbs_Ratio'] = self.df['Fiber'] / self.df['Carbohydrates']

            print("Created nutrient ratio features successfully")
            return True
        except Exception as e:
            print(f"Error creating nutrient ratios: {e}")
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

            print("Created nutrient interaction features successfully")
            return True
        except Exception as e:
            print(f"Error creating nutrient interactions: {e}")
            return False

    def normalize_features(self, columns_to_normalize=None):
        """Normalize specified features"""
        if columns_to_normalize is None:
            columns_to_normalize = self.df.select_dtypes(include=['float64', 'int64']).columns

        try:
            self.df[columns_to_normalize] = self.scaler.fit_transform(self.df[columns_to_normalize])
            print(f"Normalized {len(columns_to_normalize)} features successfully")
            return True
        except Exception as e:
            print(f"Error normalizing features: {e}")
            return False

    def save_engineered_features(self, output_path):
        """Save the engineered features to a CSV file"""
        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create filename with timestamp
            filename = f"engineered_features_{timestamp}.csv"
            full_path = os.path.join(output_path, filename)

            # Save to CSV
            self.df.to_csv(full_path, index=False)

            print(f"Saved engineered features to {full_path}")
            return full_path
        except Exception as e:
            print(f"Error saving engineered features: {e}")
            return None

def main():
    # Create output directory path
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(project_root, 'data', 'engineered')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    try:
        # Initialize feature engineer - will automatically find latest processed file
        fe = FeatureEngineer()
        
        # Create new features
        fe.create_nutrient_ratios()
        fe.create_nutrient_interactions()

        # Normalize features
        fe.normalize_features()

        # Save engineered features
        output_file = fe.save_engineered_features(output_path)

        if output_file:
            print("\nFeature engineering completed successfully!")
            print(f"Output file: {output_file}")
        else:
            print("\nFeature engineering completed with errors.")

    except Exception as e:
        print(f"\nCritical error during feature engineering: {e}")
        print("\nTroubleshooting steps:")
        print("1. Verify that data preprocessing step completed successfully")
        print(f"2. Check that processed data exists in: {os.path.join(project_root, 'data', 'processed')}")
        print("3. Ensure your processed data contains the required columns (Protein, Fat, Carbohydrates, etc.)")
        print("4. Check for NaN/infinite values in your data that might cause calculation errors")

if __name__ == "__main__":
    main()