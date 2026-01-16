import pandas as pd
import json
import os
import numpy as np  # Added numpy to handle the array conversion

# Path to the parquet file
PARQUET_FILE = 'ruscarl/data/health_bench/healthbench_train.parquet'

class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def display_sample_data():
    """
    Reads the parquet file and prints the first record in JSON format.
    """
    if not os.path.exists(PARQUET_FILE):
        print(f"[Error] File not found: {PARQUET_FILE}")
        return

    try:
        # Load the parquet file
        print(f"Loading data from: {PARQUET_FILE} ...")
        df = pd.read_parquet(PARQUET_FILE)
        
        if len(df) == 0:
            print("[Error] The dataset is empty.")
            return

        print(f"Successfully loaded. Total records: {len(df)}")
        print("-" * 60)
        print("Below is the JSON sample of the first record ")
        print("-" * 60)
        
        # Get the first row and convert it to a dictionary
        first_row = df.iloc[0].to_dict()
        
        # Print formatted JSON using the custom NumpyEncoder
        # cls=NumpyEncoder is the key fix here
        print(json.dumps(first_row, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        
        print("-" * 60)

    except Exception as e:
        print(f"[Error] An exception occurred while processing the file: {e}")

if __name__ == "__main__":
    display_sample_data()