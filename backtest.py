import pandas as pd
import random, os
import multiprocessing as mp
import numpy as np
from utils import weighted_average_per_day
from data_prep import split

from data_analysis import reg_2_filename


# Define output names and Parameters
entry_times = ["15:45", "15:50", "15:55"]
entry_thresholds = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025]
TP_thresholds = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 1]
SL_thresholds = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 1]

backtest_filename = "All_Parameters.csv"
re_run_backtest = 0 # Switch to re-run the backtest

starting_capital = 100_000

# Load prepared data

final = pd.read_csv(reg_2_filename, index_col=[0,1,2])
unique_dates = final.index.get_level_values("Date").unique().to_list()
unique_dates.sort()
train_last_date = unique_dates[int(np.floor((len(unique_dates)-1) * split[0]))]


final.loc[:, "Train/Test"] = np.where(final.index.get_level_values("Date") <= train_last_date, "Train", "Test")


# Define required functions
def find_exit_price(df, TP, SL, entry_time, SL_TP_data):
    # Function to determine the exit price of a given trade with take-profit and stop-loss
    # Checks if take-profit or stop-loss is triggered in any of the 5-min periods until the daily close
    # If neither the stop-loss nor the take-profit is triggered, the trade is exited at the closing auction at the closing price
    trade_day = df.name
    
    if df.loc["Trade_direction"] == 0:
        return 0
    elif df.loc["Trade_direction"] == -1:
        day_data = SL_TP_data.loc[pd.IndexSlice[:, trade_day],:].reset_index(level=1, drop=True)
        entry_price = df.loc["Open"]
        stop = entry_price * (1+SL)
        take_profit = entry_price*(1-TP)
        for time, _ in day_data.iterrows():
            if time < entry_time:
                continue
            else:
                high = day_data.loc[time, "High"]
                low = day_data.loc[time, "Low"]
                if (high >= stop) and (low <= take_profit):
                    random_number = random.choice([0, 1])
                    return random_number * stop + (1-random_number)*take_profit
                elif (high >= stop):
                    return stop
                elif (low <= take_profit):
                    return take_profit
    elif df.loc["Trade_direction"] == 1:
        day_data = SL_TP_data.loc[pd.IndexSlice[:, trade_day],:].reset_index(level=1, drop=True)
        entry_price = df.loc["Open"]
        stop = entry_price * (1-SL)
        take_profit = entry_price * (1+TP)
        for time, _ in day_data.iterrows():
            if time < entry_time:
                continue
            else:
                high = day_data.loc[time, "High"]
                low = day_data.loc[time, "Low"]
                if (high >= take_profit) and (low <= stop):
                    random_number = random.choice([0, 1])
                    return random_number * stop + (1-random_number)*take_profit
                elif (high >= take_profit):
                    return take_profit
                elif (low <= stop):
                    return stop
    return df.loc["Daily_close"]

def run_backtest(company, params = [], return_trades = False):
    # Function that allows to test all combinations of parameters for a given company. Through the input variable params and the flag return_trades 
    # the function also allows to retrieve all the trades for the strategy for the given set of parameters
    if params:
        times, thresholds, SLs, TPs = params
    else:
        times, thresholds, SLs, TPs = entry_times, entry_thresholds, SL_thresholds, TP_thresholds
    if not return_trades:
        output = pd.DataFrame(columns=["company", "entry_time", "entry_threshold", 
                                                "take_profit", "stop_loss", "profit_train", "profit_test", 
                                                "avg_profit_train", "avg_profit_test",  
                                                "number_of_longs_train", "number_of_longs_test", 
                                                "number_of_shorts_train", "number_of_shorts_test"])
    else:
        output = pd.DataFrame()
    data_comp = final.loc[company,:].copy()
    for entry_time in times:
        data_time = data_comp.loc[entry_time,:].copy()
        for entry_threshold in thresholds:
            # Long entries are assigned a 1, short entries a -1 and no trade is assigned a 0
            data_time.loc[:, "Trade_direction"] = np.where(data_time.loc[:, "Return_Dif"] > entry_threshold, 1 , 
                                                    np.where(data_time.loc[:, "Return_Dif"] < (-1) * entry_threshold, -1, 0))
            SL_TP_data = data_comp.loc[pd.IndexSlice[["15:45", "15:50", "15:55"], :],["High", "Low"]]
            
            for SL in SLs:
                for TP in TPs:
                        data_time.loc[:, "Exit_price"] = data_time.apply(find_exit_price, axis=1, args=(TP, SL, entry_time, SL_TP_data))
                        data_time.loc[:, "Trades"] = np.where(data_time.loc[:, "Trade_direction"] !=0, 
                                                            (data_time.loc[:, "Exit_price"] / data_time.loc[:, "Open"] - 1) * data_time.loc[:, "Trade_direction"] + 1, 1)
                        train = data_time.loc[data_time["Train/Test"] == "Train"]
                        test = data_time.loc[data_time["Train/Test"] == "Test"]

                        profit_train = train.loc[:, "Trades"].product() - 1
                        profit_test = test.loc[:, "Trades"].product() - 1
                    
                        avg_profit_train = train.loc[:, "Trades"].mean() - 1
                        avg_profit_test = test.loc[:, "Trades"].mean() - 1

                        number_of_longs_train = train.loc[train.loc[:, "Trade_direction"] == 1]["Trades"].count()
                        number_of_longs_test = test.loc[test.loc[:, "Trade_direction"] == 1]["Trades"].count()

                        number_of_shorts_train = train.loc[train.loc[:, "Trade_direction"] == -1]["Trades"].count()
                        number_of_shorts_test = test.loc[test.loc[:, "Trade_direction"] == -1]["Trades"].count()

                        if not return_trades:
                            output.loc[len(output)] = {"company": company,
                                                       "entry_time": entry_time, 
                                                       "entry_threshold": entry_threshold,
                                                       "take_profit": TP, 
                                                       "stop_loss": SL, 
                                                       "profit_train": profit_train,
                                                       "profit_test": profit_test, 
                                                       "avg_profit_train": avg_profit_train,
                                                       "avg_profit_test": avg_profit_test,
                                                       "number_of_longs_train": number_of_longs_train,
                                                       "number_of_longs_test": number_of_longs_test,
                                                       "number_of_shorts_train": number_of_shorts_train,
                                                       "number_of_shorts_test": number_of_shorts_test}
                        else:
                            out = data_time
                            out.loc[:, "Symbol"] = company
                            out.loc[:, "Time"] = entry_time
                            out.reset_index(inplace=True)
                            output = pd.concat([output, out])

    return output

if __name__ == "__main__":
    # Test various entry thresholds, take-profit and stop-loss criteria and save the results for all parameter combinations
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), backtest_filename)) or re_run_backtest:
        no_processes = mp.cpu_count()-2
        companies = sorted(final.index.get_level_values("Symbol").unique())

        pool = mp.Pool(no_processes)
        backtests = pool.map(run_backtest, companies)
        backtest_results = pd.concat(backtests)
        pool.close()
        pool.join()
        backtest_results = backtest_results.sort_values(by=["company", "profit_train"], ascending=[True, False])
        backtest_results.reset_index(drop=True, inplace=True)
        backtest_results.to_csv(backtest_filename)
        overview = backtest_results.groupby("company").head(1)
        overview.to_csv("Best_parameters_per_company.csv")

    # If the seleciton of ideal parameters is aleardy performed, load the data and then gather the performed trades of the strategy using these parameters

    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "trades_of_strategy.csv")) or 1:
        overview = pd.read_csv("Best_parameters_per_company.csv")
        trades_of_strategy = pd.DataFrame()
        # Select symbols that perform well in the training set. Here we arbitrarily choose all symbols with a return larger than 20% in the training set
        overview = overview[overview["profit_train"] > 0.2]
        for index, row in overview.iterrows():
            company = overview.loc[index, "company"]
            entry_time = overview.loc[index, "entry_time"]
            entry_threshold = overview.loc[index, "entry_threshold"]
            SL = overview.loc[index, "stop_loss"]
            TP = overview.loc[index, "take_profit"]
            resulting_trades = run_backtest(company, [[entry_time], [entry_threshold], [SL], [TP]], True)
            trades_of_strategy = pd.concat([trades_of_strategy, resulting_trades])
        

        # What percentage of Symbols are entered at what time
        share_of_times = trades_of_strategy.groupby("Time")["Symbol"].nunique() / trades_of_strategy["Symbol"].nunique()
        
        # Calculate the average profit per entry_time and date
        trades_of_strategy["Profit_per_day_time"] = trades_of_strategy.groupby(["Time", "Date"])["Trades"].transform("mean") - 1

        # Calculate the weighted average profit per date (weights are based on share of symbols per entry time)
        trades_of_strategy.loc[:, "Weights"] = trades_of_strategy["Time"].transform(lambda x: share_of_times[x])
        trades_of_strategy = trades_of_strategy.join(weighted_average_per_day(trades_of_strategy, "Profit_per_day_time", "Weights"), on="Date")

        trades_of_strategy.set_index(["Symbol", "Time", "Date"], inplace= True)
        trades_of_strategy.to_csv("trades_of_strategy.csv")

        train_performance = trades_of_strategy[trades_of_strategy["Train/Test"] == "Train"].groupby(level=2)["Profit_per_day"].mean().add(1).prod() -1
        test_performance = trades_of_strategy[trades_of_strategy["Train/Test"] == "Test"].groupby(level=2)["Profit_per_day"].mean().add(1).prod() -1
    

        print(f'Total performance training set: {round(train_performance*100, 2)}')
        print(f'Total performance test set: {round(test_performance*100, 2)}')


    

    



    
    


