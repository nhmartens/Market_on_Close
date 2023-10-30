import pandas as pd
import os

def consolidate(df, symbol, start_time, end_time):
    def cons(df):
        times, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        symbols = 6 * [symbol]

        for time in ["15:45", "15:50", "15:55", "16:00", "16:05"]:
            times.append(time)
            [h, m] = [int(element) for element in time.split(":")]
            view = df.loc[(df.index.hour==h) & (df.index.minute==m)]
            if not view.empty:
                opens.append(view.Open.values[0])
                highs.append(view.High.values[0])
                lows.append(view.Low.values[0])
                closes.append(view.Close.values[0])
                volumes.append(view.Volume.values[0])
            else:
                opens.append("NA")
                highs.append("NA")
                lows.append("NA")
                closes.append("NA")
                volumes.append("NA")
        times.append('Day')
        opens.append(df.loc[df.index.min()]["Open"])
        highs.append(df["High"].max())
        lows.append(df["Low"].min())
        daily_close = closes[-2] if closes[-2] != "NA" else closes[-1]
        closes.append(daily_close)
        volumes.append(df['Volume'].sum())
        
        return pd.DataFrame({"Symbol": symbols,
                             "Time": times,
                             "Open": opens, 
                             "High": highs, 
                             "Low": lows, 
                             "Close": closes,
                             "Volume": volumes
                             })
    
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    out_df = df.between_time(start_time, end_time)
    out_df = out_df.groupby(out_df.index.date).apply(lambda x: cons(x)).reset_index(names = ["Date", "drop"])
    out_df.drop(["drop"], axis=1, inplace = True)
    out_df.set_index(["Symbol", "Time", "Date"], inplace= True)
    return out_df



def consolidate_new(filepath, start_time="9:30", end_time="16:05"):
    symbol = os.path.split(filepath)[1].split("_")[0].replace(" #F", "_F")
    print(symbol)
    "/Users/nilsmartens/Documents/Code Repositories/Market_on_Close/Data/Futures/ALI #F_5.csv"
    df = pd.read_csv(filepath)
    def cons(df):
        times, opens, highs, lows, closes, volumes, returns = [], [], [], [], [], [], []
        symbols = 6 * [symbol]

        for time in ["15:45", "15:50", "15:55", "16:00", "16:05"]:
            times.append(time)
            [h, m] = [int(element) for element in time.split(":")]
            view = df.loc[(df.index.hour==h) & (df.index.minute==m)]
            if not view.empty:
                returns.append(1)
                opens.append(view.Open.values[0])
                highs.append(view.High.values[0])
                lows.append(view.Low.values[0])
                closes.append(view.Close.values[0])
                volumes.append(view.Volume.values[0])
            else:
                returns.append("NA")
                opens.append("NA")
                highs.append("NA")
                lows.append("NA")
                closes.append("NA")
                volumes.append("NA")
        times.append('Day')
        opens.append(df.loc[df.index.min()]["Open"])
        returns = [closes[i] / opens[-1] -1 if r != "NA" else "NA" for i, r in enumerate(returns)]
        highs.append(df["High"].max())
        lows.append(df["Low"].min())
        daily_close = closes[-2] if closes[-2] != "NA" else closes[-1] if closes[-1] != "NA" else closes[-3]
        returns.append(opens[-1] / daily_close -1 if opens[-1] != "NA" and daily_close != "NA" else "NA")
        closes.append(daily_close)
        volumes.append(df['Volume'].sum())
        
        return pd.DataFrame({"Symbol": symbols,
                             "Time": times,
                             "Open": opens, 
                             "High": highs, 
                             "Low": lows, 
                             "Close": closes,
                             "Volume": volumes,
                             "Return": returns
                             })
    
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    out_df = df.between_time(start_time, end_time)
    out_df = out_df.groupby(out_df.index.date).apply(lambda x: cons(x)).reset_index(names = ["Date", "drop"])
    out_df.drop(["drop"], axis=1, inplace = True)
    out_df.set_index(["Symbol", "Time", "Date"], inplace= True)
    return out_df


    
