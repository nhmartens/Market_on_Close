#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance

trades = pd.read_csv("trades_of_strategy.csv")
trades['Date'] = pd.to_datetime(trades['Date'])

spx = raw_data = yfinance.download (tickers = "^SPX", start = "2015-01-01", 
                              end = "2022-06-01", interval = "1d")



trades[trades["Train/Test"] == "Train"].groupby("Date")["Profit_per_day"].mean().add(1).cumprod().plot(title="Training Set Performance")
plt.show()
plt.savefig(f"./Plots/Training Set Performance.jpg")
joined = trades[trades["Train/Test"] == "Train"].groupby("Date")[["Profit_per_day"]].mean()
joined.loc[:, "MOC_Strategy"] = joined.loc[:, "Profit_per_day"].add(1).cumprod()
#joined = trades[trades["Train/Test"] == "Train"].groupby("Date")[["Profit_per_day"]].mean().add(1).cumprod()
joined = joined.join(spx)
joined.loc[:, "Return"] = joined.loc[:, "Adj Close"].pct_change()
joined.loc[:, "SPX"] = joined.loc[:, "Return"].add(1).cumprod()
joined.loc[:,["MOC_Strategy", "SPX"]].plot()
plt.savefig(f"./Plots/Training Set Performance vs SPX.jpg")
plt.show()
print(f"Sharpe Ratio MOC_Strategy Training Set: {np.sqrt(252) * joined.loc[:,'Profit_per_day'].mean() / (joined.loc[:,'Profit_per_day'].std())}")
print(f"Sharpe Ratio SPX Training Set: {np.sqrt(252) * joined.loc[:,'Return'].mean() / (joined.loc[:,'Return'].std())}")

trades[trades["Train/Test"] == "Test"].groupby("Date")["Profit_per_day"].mean().add(1).cumprod().plot(title="Test Set Performance")
plt.show()
plt.savefig(f"./Plots/Test Set Performance.jpg")
joined = trades[trades["Train/Test"] == "Test"].groupby("Date")[["Profit_per_day"]].mean()
joined.loc[:, "MOC_Strategy"] = joined.loc[:, "Profit_per_day"].add(1).cumprod()
joined = joined.join(spx)
joined.loc[:, "Return"] = joined.loc[:, "Adj Close"].pct_change()
joined.loc[:, "SPX"] = joined.loc[:, "Return"].add(1).cumprod()
#joined.rename(columns={"Profit_per_day": "MOC_Strategy"}, inplace=True)
joined.loc[:,["MOC_Strategy", "SPX"]].plot()
plt.savefig(f"./Plots/Test Set Performance vs SPX.jpg")
plt.show()
print(f"Sharpe Ratio MOC_Strategy Training Set: {np.sqrt(252) * joined.loc[:,'Profit_per_day'].mean() / (joined.loc[:,'Profit_per_day'].std())}")
print(f"Sharpe Ratio SPX Training Set: {np.sqrt(252) * joined.loc[:,'Return'].mean() / (joined.loc[:,'Return'].std())}")




# %%
trades = pd.read_csv("trades_of_strategy.csv")
trades['Date'] = pd.to_datetime(trades['Date'])
symbols = trades["Symbol"].unique()
trades.set_index("Date", inplace=True)
for symbol in symbols:
    trades[(trades["Symbol"] == symbol) & (trades["Train/Test"] == "Train")]["Trades"].cumprod().plot(title=f"Training Set Performance of {symbol}")
    plt.show()
    plt.savefig(f"./Plots/Stocks_Training/Training Set Performance {symbol}.jpg")
    trades[(trades["Symbol"] == symbol) & (trades["Train/Test"] == "Test")]["Trades"].cumprod().plot(title=f"Test Set Performance of {symbol}")
    plt.savefig(f"./Plots/Stocks_Test/Test Set Performance {symbol}.jpg")
    plt.show()
    

# %%
trades = pd.read_csv("trades_of_strategy.csv")
trades["Date"] = pd.to_datetime(trades["Date"])
test_first_day = trades[trades["Train/Test"] == "Test"]["Date"].min()
trades.set_index("Date", inplace=True)
trades.loc[:, "No_trades"] = trades.loc[:, "Trade_direction"].abs()
trades.groupby(level=0)["No_trades"].sum().cumsum().plot(title="Number of Trades over Time")
plt.axvline(x=test_first_day, color='red', linestyle='--', label='Test Set')
plt.savefig(f"./Plots/Number of Trades over Time.jpg")



# %%
trades = pd.read_csv("trades_of_strategy.csv")
trades["Date"] = pd.to_datetime(trades["Date"])
trades.loc[:, "No_shorts"] = np.where(trades.loc[:, "Trade_direction"] == -1, 1, 0)
trades.loc[:, "No_longs"] = np.where(trades.loc[:, "Trade_direction"] == 1, 1, 0)

trades.groupby("Date")[["No_shorts", "No_longs"]].sum().cumsum().plot(title="Longs vs Shorts")
plt.axvline(x=test_first_day, color='red', linestyle='--', label='Test Set')
plt.savefig(f"./Plots/Number of LongsShorts over Time.jpg")
plt.show()


grouped = trades.groupby("Date")[["No_shorts", "No_longs"]].sum()
grouped.loc[:, "Longs-Shorts"] = grouped.loc[:, "No_longs"] - grouped.loc[:, "No_shorts"]
grouped.loc[:, "Longs-Shorts"].plot(title="Number of Longs minus Shorts")
plt.savefig(f"./Plots/Number of Longs minus Shorts over Time.jpg")
plt.show()

# %%
