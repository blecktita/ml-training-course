"""
This module downloads a dataset from a provided URL, saves it to a local file,
and loads it into a pandas DataFrame for further processing.
"""

import os
import sys
import requests
import pandas as pd

# Create the 'data' folder if it doesn't exist
DATA_FOLDER = 'data'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

if len(sys.argv) < 3:
    print("Please provide the data URL and filename as command-line arguments.")
    print("Usage: python script.py <data_url> <filename>")
    sys.exit(1)

DATA_URL = sys.argv[1]
FILENAME = sys.argv[2]

# Download the data
response = requests.get(DATA_URL, timeout=10)

# Save the data to a file in the 'data' folder
data_file = os.path.join(DATA_FOLDER, FILENAME)
with open(data_file, 'wb') as file:
    file.write(response.content)

print(f"Data saved to: {data_file}")

# Load the data into a pandas DataFrame
df = pd.read_csv(data_file)

# Print the first few rows of the DataFrame
print(df.head())
