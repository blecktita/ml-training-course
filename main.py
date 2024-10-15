import pandas as pd
import yaml
from src.DataLoader import DataLoader


# Configure path to data: 
with open('src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_store = config['data']['original_data']
sep = config['delimiters']['sep']


def main():
    # Load the dataset
    data_loader = DataLoader(data_store, sep)
    df = data_loader.load_data()
    print(df.shape)
    print(df.columns)

if __name__ == "__main__":
    main()
