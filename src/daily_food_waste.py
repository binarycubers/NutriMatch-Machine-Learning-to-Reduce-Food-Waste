import pandas as pd
import numpy as np
import os
from datetime import datetime

class DailyFoodWasteCalculator:
    def __init__(self, base_dir=None):
        """
        Initialize the DailyFoodWasteCalculator class
        base_dir: Base directory for all data operations (should be your project root)
        """
        # Set project root directory
        self.base_dir = base_dir if base_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Set up log file
        self.log_file = os.path.join(self.base_dir, 'logs',
                                   f'daily_food_waste_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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
        self.daily_waste_df = None
        self.original_shape = None

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

    def load_processed_data(self, file_name):
        """
        Load the processed dataset
        file_name: Name of the processed data file
        """
        try:
            # First try the processed directory
            source_path = os.path.join(self.data_dirs['processed'], file_name)
            
            # If not found in processed directory, try the root directory
            if not os.path.exists(source_path):
                source_path = os.path.join(self.base_dir, file_name)
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"File not found at: {source_path}")

            self.log_message(f"Loading data from: {source_path}")

            # Read CSV with dayfirst=True to handle day/month format
            self.df = pd.read_csv(source_path, parse_dates=['Date'], dayfirst=True)
            self.original_shape = self.df.shape

            self.log_message(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error loading data: {str(e)}")
            return False

    def calculate_daily_food_waste(self):
        """Calculate average daily food waste for each nutrient"""
        try:
            # List of nutrient columns (adjust these based on your actual columns)
            nutrient_columns = [col for col in self.df.columns 
                               if col not in ['Date', 'Quantity'] and 
                               self.df[col].dtype in [np.float64, np.int64]]

            self.log_message("Starting daily food waste calculation...")
            self.log_message(f"Nutrient columns being processed: {nutrient_columns}")

            # Group by Date and calculate mean for nutrient columns
            self.daily_waste_df = self.df.groupby('Date')[nutrient_columns].mean().reset_index()

            # Add sum of quantities per day if Quantity column exists
            if 'Quantity' in self.df.columns:
                daily_quantities = self.df.groupby('Date')['Quantity'].sum().reset_index()
                self.daily_waste_df['Total_Quantity'] = daily_quantities['Quantity']

            # Sort by date
            self.daily_waste_df = self.daily_waste_df.sort_values('Date')

            self.log_message(f"Daily food waste calculated. Shape: {self.daily_waste_df.shape}")
            self.log_message(f"Date range: {self.daily_waste_df['Date'].min()} to {self.daily_waste_df['Date'].max()}")

            return True
        except Exception as e:
            self.log_message(f"Error calculating daily food waste: {str(e)}")
            return False

    def save_daily_waste_data(self, file_name='daily_food_waste.csv'):
        """Save the daily food waste dataset"""
        try:
            if self.daily_waste_df is None:
                raise ValueError("No daily waste data to save. Run calculate_daily_food_waste first.")

            # Ensure processed directory exists
            os.makedirs(self.data_dirs['processed'], exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name_with_timestamp = f"{os.path.splitext(file_name)[0]}_{timestamp}{os.path.splitext(file_name)[1]}"
            save_path = os.path.join(self.data_dirs['processed'], file_name_with_timestamp)

            self.daily_waste_df.to_csv(save_path, index=False)
            self.log_message(f"Daily food waste data saved to {save_path}")

            # Save summary statistics
            stats_path = os.path.join(self.data_dirs['processed'],
                                    f"daily_waste_stats_{timestamp}.txt")

            with open(stats_path, 'w') as f:
                f.write("Daily Food Waste Statistics\n")
                f.write("==========================\n\n")
                f.write(f"Total days: {len(self.daily_waste_df)}\n")
                f.write(f"Date range: {self.daily_waste_df['Date'].min()} to {self.daily_waste_df['Date'].max()}\n\n")
                f.write("Average daily waste by nutrient:\n")
                for column in self.daily_waste_df.columns:
                    if column != 'Date':
                        f.write(f"{column}: {self.daily_waste_df[column].mean():.2f}\n")

            self.log_message(f"Statistics saved to {stats_path}")
            return save_path
        except Exception as e:
            self.log_message(f"Error saving daily waste data: {str(e)}")
            return None

def main():
    # Initialize calculator with the project root directory
    base_dir = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste"  # Your project root directory
    calculator = DailyFoodWasteCalculator(base_dir)

    # Print directory structure for debugging
    print("\nCurrent directory structure:")
    for dir_type, dir_path in calculator.data_dirs.items():
        print(f"{dir_type.upper():<10}: {dir_path}")

    # Specify the input file name (now checks both processed dir and root dir)
    input_file = 'processed_data_20250506_223522.csv'  # Your input file name

    # Load processed data
    data_loaded = calculator.load_processed_data(input_file)

    if not data_loaded:
        print("\nFailed to load processed data. Possible solutions:")
        print(f"1. Ensure {input_file} exists in either:")
        print(f"   - {calculator.data_dirs['processed']}")
        print(f"   - {calculator.base_dir}")
        print("2. Check the file permissions")
        print("3. Verify the file is not open in another program")
        print("4. Check that the date format in your CSV is day/month/year")
        return

    # Calculate daily food waste
    if calculator.calculate_daily_food_waste():
        # Save daily waste data
        daily_waste_path = calculator.save_daily_waste_data('daily_food_waste.csv')
        if daily_waste_path:
            print(f"\nSuccessfully saved daily food waste data to: {daily_waste_path}")

if __name__ == "__main__":
    main()