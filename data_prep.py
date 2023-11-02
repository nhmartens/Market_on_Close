
import pandas as pd
import numpy as np
import os
from utils import consolidate, consolidate_new
import multiprocessing as mp

# Split the data in train and test set
split = [0.9, 0.1]

dir = os.path.dirname(os.path.realpath(__file__))
stock_data_dir = os.path.join(dir, "Data", "Stocks")
future_data_dir = os.path.join(dir, "Data", "Futures")

# Extract all available company tickers from file names (e.g. "AAPL" from "AAPL_5.csv")
company_tickers = [name.replace("_5.csv","") for name in os.listdir(stock_data_dir) if name[-4:] == ".csv"]
# Extract all available future tickers form file names
future_tickers = [name.replace(" #F_5.csv","_F") for name in os.listdir(future_data_dir) if name[-4:] == ".csv"]

# Use multiprocessing to gather the required information from csv-files with 5-minute data into one multi-index dataframe
if __name__ == '__main__':
    no_processes = mp.cpu_count()-3
    stock_files = [os.path.join(stock_data_dir, filename) for filename in os.listdir(stock_data_dir) if filename.endswith(".csv")]
    future_files = [os.path.join(future_data_dir, filename) for filename in os.listdir(future_data_dir) if filename.endswith(".csv")]
    files = stock_files + future_files
    pool = mp.Pool(no_processes)
    processed_dfs = pool.map(consolidate_new, files)

    df = pd.concat(processed_dfs)
    pool.close()
    pool.join()
    unique_dates = df.loc[:, "Date"].unique()
    unique_dates.sort()
    train_last_date = unique_dates[int(np.floor((len(unique_dates)-1) * split[0]))]

    train = df[df["Date"] <= train_last_date].set_index(["Symbol", "Time", "Date"]).sort_index()
    test = df[(df["Date"] > train_last_date)].set_index(["Symbol", "Time", "Date"]).sort_index()


    train.to_csv("train.csv")
    test.to_csv("test.csv")

