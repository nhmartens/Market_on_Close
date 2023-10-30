import pandas as pd
import numpy as np
import os, random
import multiprocessing as mp
from itertools import product
from itertools import combinations
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from data_prep import company_tickers, future_tickers


# Load prepared data
df = pd.read_csv("df.csv", index_col=[0,1,2])
reg_data = df.loc[pd.IndexSlice[:, "Day", :], "Return"].reset_index(level="Time", drop = True).dropna()
roll_data = df[df.index.get_level_values('Time').isin(["15:45", "15:50", "15:55", "16:00"])]
future_tickers = [ticker for ticker in future_tickers if ticker in reg_data.index.get_level_values("Symbol")]

# Check number of vaild data points for every remaining future ticker - removed if less than 300
for ticker in future_tickers:
    if reg_data.loc[ticker].shape[0] < 100:
        future_tickers.remove(ticker)

# Perform all possible combinations of regressions from daily stock returns on one or two future returns
# The aim is to select the pairs with the highest explanatory power (Selection based on R2)
def prep_data_for_OLS(combination, df_merge = False):
    (company, (future1, future2)) = combination
    if not df_merge:
        if future1 == "NA" or future2 == "NA":
            future1 = future1 if future2 == "NA" else future2
            future2 = "NA"
            common_days = list(set(reg_data.loc[company, :].index.get_level_values("Date"))
                            .intersection(reg_data.loc[future1, :].index.get_level_values("Date")))
            if not common_days:
                print(f"{combination} no common days")
                return
            common_days.sort()
            X = reg_data.loc[(future1, common_days)].to_numpy()
        
        else:
            common_days = list(set(
                reg_data.loc[company, :].index.get_level_values("Date"))
                .intersection(reg_data.loc[future1, :].index.get_level_values("Date"))
                .intersection(reg_data.loc[future2, :].index.get_level_values("Date"))
                )
            if not common_days:
                print(f"{combination} no common days")
                return
            common_days.sort()
            X = np.column_stack([reg_data.loc[(future1, common_days)].to_numpy(), 
                                reg_data.loc[(future2, common_days)].to_numpy()])
            
        y = reg_data.loc[(company, common_days)].to_numpy()
        return y, sm.add_constant(X), company, future1, future2
    
    else:
        if future1 == "NA" or future2 == "NA":
            out = roll_data.loc[company].join(
                  roll_data.loc[future1]['Return'], on=['Time','Date'],rsuffix='_future1', how="inner")
            out.loc[:,"Return_future2"] = [0] * out.shape[0]
        else:
            out = roll_data.loc[company].join(
                roll_data.loc[future1]['Return'], on=['Time','Date'],rsuffix='_future1', how="inner").join(
                roll_data.loc[future2]['Return'], on=['Time','Date'],rsuffix='_future2', how="inner")
        return out
        
    

def run_ols(combination):
    y, X, company, future1, future2 = prep_data_for_OLS(combination)
    result = sm.OLS(y, X).fit()
    out = pd.DataFrame({"Symbol": company,
                        "Intercept": result.params[0],
                        "Beta_1": result.params[1], 
                        "Beta_2": result.params[2] if len(result.params) > 2 else "NA",
                        "Future_1": future1,
                        "Future_2": future2,
                        "R2": result.rsquared,
                        "R2_adj": result.rsquared_adj}) 
    return out

def run_rolling_OLS(combination):
    data = prep_data_for_OLS(combination, True)
    data.dropna(subset=["Return", "Return_future1", "Return_future2"], inplace = True)
    output = pd.DataFrame()
    for time in ["15:45", "15:50", "15:55", "16:00"]:
        time_df = data.loc[time]
        if time == "16:00":
            out = pd.DataFrame({"Time": [time] * time_df.shape[0]}, index = time_df.index)
            out.loc[:, "const"] = 0
            out.loc[:, "Return_future1"] = 0
            out.loc[:, "Return_future2"] = 0
            out.loc[:, "Daily_close"] = data.loc["16:00"]["Close"]
        else:
            y = time_df.loc[:,"Return"]
            X = time_df.loc[:, ["Return_future1", "Return_future2"]]
            # returns estimates for every day for a specific time and specific symbol
            results = RollingOLS(y, sm.add_constant(X), 100).fit(params_only=True)
            out = results.params
            out = out.shift(1)
            out.loc[:, "Time"] = [time] * out.shape[0]
            out = out.join(data.loc["16:00"]["Close"], on = "Date", how="inner")
            
            out.rename(columns={"Close": "Daily_close"}, inplace=True)
        output = pd.concat([output, out])
    #output.loc[:,"Symbol"] = combination[0]
    output.set_index(["Time", output.index], inplace=True)
    data = data.join(output, on=["Time", "Date"], rsuffix="_beta")
    data.loc[:,"Pred_Return"] = (data["const"] + 
                                 data["Return_future1_beta"] * data["Return_future1"] + 
                                 data["Return_future2_beta"] * data["Return_future2"] )
    data.loc[:,"Return_Dif"] = data["Pred_Return"] - data["Return"]
    data.loc[:, "Symbol"] = combination[0]
    data.dropna(subset=["Return_Dif", "Daily_close"], inplace = True)
    data.reset_index(inplace=True)
    data.set_index(["Symbol", "Time", "Date"], inplace=True)
    return data



if __name__ == "__main__":
    # Run first regression: Intraday stock returns on intraday future returns (09:30 until 16:00):
    reg_1_filename = "Regression.csv"
    reg_2_filename = "Rolling_Reg.csv"
    no_processes = mp.cpu_count()-2
    future_tickers.append("NA")
    future_combinations = [tuple(sorted(comb)) for comb in combinations(future_tickers, 2)]
    combs = list(product(company_tickers, future_combinations))
    # Switch to re-run the regressions
    re_run_regressions = 0

    # Only re-run the regressions if the csv containing the results does not exist already or the switch indicates to
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), reg_1_filename)) or re_run_regressions:
        pool = mp.Pool(no_processes)
        #ran_regressions = pool.map(run_ols, combs)
        with mp.Pool(no_processes) as pool, tqdm(total=len(combs)) as pbar:
            ran_regressions = list(tqdm(pool.imap(run_ols, combs), total=len(combs), position=0))
        reg_res = pd.concat(ran_regressions)
        pool.close()
        pool.join()
        reg_res.sort_values(by="R2_adj", inplace=True)
        reg_res.to_csv(reg_1_filename)

    
    # Run second regression: Rolling regression of intraday stock returns on intraday future returns (e.g. 09:30 until 15:45) with a window of 100 days
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), reg_2_filename)) or re_run_regressions:
        regs = pd.read_csv(reg_1_filename, header = 0, names=["Symbol", "Intercept", "Beta_1", "Beta_2", "Future_1", "Future_2", "R2", "R2_adj"])
        idx = regs.groupby("Symbol")["R2_adj"].idxmax()
        regs = regs.loc[idx]
        regs = regs.loc[regs["R2_adj"] > 0.2]
        regs = regs.sort_values(by="R2_adj", ascending=False).reset_index(drop=True)
        combinations = [(row["Symbol"], (row["Future_1"], row["Future_2"])) for _, row in regs.iterrows()]
        pool = mp.Pool(no_processes)
        processed_dfs = pool.map(run_rolling_OLS, combinations)
        final = pd.concat(processed_dfs)
        pool.close()
        pool.join()
        
        final.to_csv(reg_2_filename)


    # Test various entry thresholds, take-profit and stop-loss criteria

    final = pd.read_csv(reg_2_filename, index_col=[0,1,2])
 
    companies = sorted(final.index.get_level_values("Symbol").unique())
    entry_times = ["15:45", "15:50", "15:55"]
    entry_thresholds = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025]
    TP_thresholds = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03]
    SL_thresholds = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025]

    #final.loc[:, "Trades_optimal_strategy"] = "NA"

    #print(final.loc[pd.IndexSlice["AUY", ["15:45", "15:50"], "2015-07-07"],["High", "Low"]].reset_index(level=[0,2], drop=True))
    #print(final.index)

    def find_exit_price(df, TP, SL, entry_time):
        trade_day = df.name
        
        if df.loc["Trade_direction"] == 0:
            exit_price = 0
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
                        exit_price = random_number * stop + (1-random_number)*take_profit
                        return random_number * stop + (1-random_number)*take_profit
                    elif (high >= stop):
                        return stop
                    elif (low <= take_profit):
                        return take_profit
        elif df.loc["Trade_direction"] == 1:
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
                    if (high >= take_profit) and (low <= stop):
                        random_number = random.choice([0, 1])
                        exit_price = random_number * stop + (1-random_number)*take_profit
                        return random_number * stop + (1-random_number)*take_profit
                    elif (high >= take_profit):
                        return take_profit
                    elif (low <= stop):
                        return stop
        return df.loc["Daily_close"]
        


    companies = ["HMY"]
    entry_times = ["15:45"]
    entry_thresholds = [0.005]
    parameters_results = pd.DataFrame(columns=["entry_time", "entry_threshold", "take_profit", "stop_loss", "profit", "avg_profit"])

    for company in companies:
        data_comp = final.loc[company,:]
        highest_profit = 0

        for entry_time in entry_times:
            data_time = data_comp.loc[entry_time,:]
            for entry_threshold in entry_thresholds:
                # Long entries are assigned a 1, short entries a -1 and no trade is assigned a 0
                data_time.loc[:, "Trade_direction"] = np.where(data_time.loc[:, "Return_Dif"] > entry_threshold, 1 , 
                                                     np.where(data_time.loc[:, "Return_Dif"] < (-1) * entry_threshold, -1, 0))
                SL_TP_data = data_comp.loc[pd.IndexSlice[["15:45", "15:50", "15:55"], :],["High", "Low"]]
                
                for SL in SL_thresholds:
                    for TP in TP_thresholds:
                        data_time.loc[:, "Exit_price"] = data_time.apply(find_exit_price, axis=1, args=(TP, SL, entry_time))
                        data_time.loc[:, "Trades"] = 1
                        #data_time.loc[data_time.loc[:, "Trade_direction"] != 0, "Trades"] = (data_time.loc[:, "Exit_price"] / data_time.loc[:, "Open"] - 1) * data_time.loc[:, "Trade_direction"] + 1
                        data_time.loc[:, "Trades"] = np.where(data_time.loc[:, "Trade_direction"] !=0, 
                                                            (data_time.loc[:, "Exit_price"] / data_time.loc[:, "Open"] - 1) * data_time.loc[:, "Trade_direction"] + 1, 1)
                        profit = data_time.loc[:, "Trades"].product() - 1
                        avg_profit = data_time.loc[:, "Trades"].mean() - 1
                        parameters_results.loc[len(parameters_results)] = {"entry_time": entry_time, "entry_threshold": entry_threshold, "take_profit": TP, "stop_loss": SL, "profit": profit, "avg_profit": avg_profit}
                        if profit > highest_profit:
                            data_time.to_csv("test.csv")
    print(data_time.head())
    print(parameters_results)





                # 
                




    
    
