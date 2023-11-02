import pandas as pd
import numpy as np
import os, random
import multiprocessing as mp
from itertools import product
from itertools import combinations
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from data_prep import company_tickers, future_tickers, split


# Define output names
re_run_regressions = 1 # Switch to re-run the regressions
reg_1_filename = "Regression.csv"
reg_2_filename = "Rolling_Reg.csv"

train_set_filename = "train.csv"
test_set_filename = "test.csv"


# Load prepared data
df = pd.read_csv(train_set_filename, index_col=[0,1,2])
df_test = pd.read_csv(test_set_filename, index_col=[0,1,2])
df_total = pd.concat([df, df_test]).sort_index()

reg_data = df.loc[pd.IndexSlice[:, "Day", :], "Return"].reset_index(level="Time", drop = True).dropna()
roll_data = df_total[df_total.index.get_level_values("Time").isin(["15:45", "15:50", "15:55", "16:00"])]
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
                        "R2_adj": result.rsquared_adj}, index=[0]) 
    return out

def run_rolling_OLS(combination):
    print(combination)
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
    no_processes = mp.cpu_count()-1
    future_tickers.append("NA")
    future_combinations = [tuple(sorted(comb)) for comb in combinations(future_tickers, 2)]
    combs = list(product(company_tickers, future_combinations))
    
 

    # Only re-run the regressions if the csv containing the results does not exist already or the switch indicates to
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), reg_1_filename)) or re_run_regressions:
        pool = mp.Pool(no_processes)
        with mp.Pool(no_processes) as pool, tqdm(total=len(combs)) as pbar:
            ran_regressions = list(tqdm(pool.imap(run_ols, combs), total=len(combs), position=0))
        reg_res = pd.concat(ran_regressions)
        pool.close()
        pool.join()
        reg_res.sort_values(by="R2_adj", inplace=True)
        reg_res.reset_index(inplace=True)
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
        

    
    
