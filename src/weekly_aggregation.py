import pandas as pd
import os
from datetime import datetime

class WeeklyAggregator:
    def __init__(self, base_dir=None):
        """
        Initialize the WeeklyAggregator class
        base_dir: Base directory for all data operations (should be your project root)
        """
        # Set project root directory
        self.base_dir = base_dir if base_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Set up log file
        self.log_file = os.path.join(self.base_dir, 'logs',
                                   f'weekly_aggregation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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
        self.weekly_df = None
        self.agg_method = None

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

    def load_daily_data(self, file_name='daily.csv'):
        """
        Load the daily dataset
        file_name: Name of the daily data file
        """
        try:
            # Check multiple possible locations
            possible_paths = [
                os.path.join(self.data_dirs['processed'], file_name),
                os.path.join(self.base_dir, file_name)
            ]
            
            source_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    source_path = path
                    break
            
            if not source_path:
                raise FileNotFoundError(f"File not found in: {possible_paths}")

            self.log_message(f"Loading daily data from: {source_path}")

            # Read CSV with proper date parsing
            self.df = pd.read_csv(source_path, parse_dates=['Date'])
            
            # Validate required columns
            if 'Date' not in self.df.columns:
                raise ValueError("Data must contain 'Date' column")
                
            # Ensure Date column is in datetime format
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            self.log_message(f"Daily data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error loading daily data: {str(e)}")
            return False

    def aggregate_weekly(self, agg_method='sum', exclude_cols=None):
        """
        Aggregate daily data to weekly data
        agg_method: 'sum' or 'mean'
        exclude_cols: List of columns to exclude from aggregation
        """
        try:
            self.log_message("Starting weekly aggregation...")
            self.agg_method = agg_method

            # Validate aggregation method
            if agg_method not in ['sum', 'mean']:
                raise ValueError("Invalid aggregation method. Use 'sum' or 'mean'")

            # Extract ISO calendar week components
            self.df['Year'] = self.df['Date'].dt.isocalendar().year
            self.df['Week'] = self.df['Date'].dt.isocalendar().week

            # Identify columns for aggregation
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            default_exclusions = ['Year', 'Week']
            exclude_cols = exclude_cols or []
            cols_to_agg = [col for col in numeric_cols 
                          if col not in default_exclusions + exclude_cols]

            # Group by week and aggregate
            agg_func = {'sum': sum, 'mean': 'mean'}[agg_method]
            self.weekly_df = self.df.groupby(['Year', 'Week'])[cols_to_agg].agg(agg_func).reset_index()

            # Calculate week start/end dates using ISO week definition
            self.weekly_df['Week_Start'] = pd.to_datetime(
                self.weekly_df['Year'].astype(str) + '-' +
                self.weekly_df['Week'].astype(str) + '-1', 
                format='%Y-%W-%w'
            )
            self.weekly_df['Week_End'] = self.weekly_df['Week_Start'] + pd.Timedelta(days=6)

            # Reorder columns for better readability
            column_order = ['Year', 'Week', 'Week_Start', 'Week_End'] + \
                         [col for col in self.weekly_df.columns 
                          if col not in ['Year', 'Week', 'Week_Start', 'Week_End']]
            self.weekly_df = self.weekly_df[column_order]

            self.log_message(f"Weekly aggregation completed. Shape: {self.weekly_df.shape}")
            return True
        except Exception as e:
            self.log_message(f"Error in weekly aggregation: {str(e)}")
            return False

    def save_weekly_data(self, file_name='weekly_food_waste.csv'):
        """Save the weekly aggregated dataset"""
        try:
            if self.weekly_df is None:
                raise ValueError("No weekly data to save. Run aggregate_weekly first.")

            # Ensure output directory exists
            os.makedirs(self.data_dirs['processed'], exist_ok=True)

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name_with_ts = f"{os.path.splitext(file_name)[0]}_{timestamp}.csv"
            save_path = os.path.join(self.data_dirs['processed'], file_name_with_ts)

            # Save main data file
            self.weekly_df.to_csv(save_path, index=False)
            self.log_message(f"Weekly data saved to {save_path}")

            # Generate comprehensive statistics
            stats_path = os.path.join(self.data_dirs['processed'],
                                    f"weekly_stats_{timestamp}.txt")
            self._generate_statistics_file(stats_path)

            return save_path
        except Exception as e:
            self.log_message(f"Error saving weekly data: {str(e)}")
            return None

    def _generate_statistics_file(self, stats_path):
        """Generate detailed statistics report"""
        with open(stats_path, 'w') as f:
            f.write("Weekly Food Waste Statistics Report\n")
            f.write("==================================\n\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {self.df.shape[0]} daily records\n")
            f.write(f"Aggregation method: {self.agg_method}\n\n")
            
            f.write("Temporal Coverage:\n")
            f.write(f"- First week: {self.weekly_df['Week_Start'].min().date()} "
                   f"to {self.weekly_df['Week_End'].min().date()}\n")
            f.write(f"- Last week:  {self.weekly_df['Week_Start'].max().date()} "
                   f"to {self.weekly_df['Week_End'].max().date()}\n")
            f.write(f"- Total weeks: {len(self.weekly_df)}\n\n")
            
            f.write("Aggregated Metrics Summary:\n")
            for col in self.weekly_df.columns:
                if col not in ['Year', 'Week', 'Week_Start', 'Week_End']:
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean:    {self.weekly_df[col].mean():.2f}\n")
                    f.write(f"  Median:  {self.weekly_df[col].median():.2f}\n")
                    f.write(f"  Std Dev: {self.weekly_df[col].std():.2f}\n")
                    f.write(f"  Minimum: {self.weekly_df[col].min():.2f}\n")
                    f.write(f"  Maximum: {self.weekly_df[col].max():.2f}\n")
                    f.write(f"  Total:   {self.weekly_df[col].sum():.2f}\n")

            f.write("\nNote: All dates follow ISO week numbering system (Monday as first day of week)\n")

        self.log_message(f"Detailed statistics saved to {stats_path}")

def main():
    # Initialize aggregator with project directory
    base_dir = r"D:\NutriMatch-Machine-Learning-to-Reduce-Food-Waste"
    aggregator = WeeklyAggregator(base_dir)

    # Print directory structure for verification
    print("\nCurrent directory structure:")
    for dir_type, dir_path in aggregator.data_dirs.items():
        print(f"{dir_type.upper():<10}: {dir_path}")

    # Load daily data with flexible location handling
    if not aggregator.load_daily_data('daily_food_waste_20250506_234610.csv'):
        print("\nFailed to load daily data. Possible issues:")
        print("- File not found in processed directory or project root")
        print("- Invalid date format in CSV")
        print("- Missing required columns")
        return

    # Perform weekly aggregation (can change to 'mean' if needed)
    if aggregator.aggregate_weekly(agg_method='sum', exclude_cols=[]):
        # Save results with comprehensive outputs
        output_path = aggregator.save_weekly_data()
        if output_path:
            print(f"\nSuccess! Weekly data saved to:")
            print(f"Main data: {output_path}")
            print(f"Statistics: {output_path.replace('.csv', '_stats.txt')}")

if __name__ == "__main__":
    main()