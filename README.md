# Market on Close

## General intuition
The idea behind the Market-on-Close strategy is to find stocks whose return's depend on the prices of some futures (e.g. gold mining stocks and gold futures). The assumption is that any price deviances (from the ordinary daily relationship of stock and future return) that occur within the trading day are likely to dissappear with the closing auction (due to rebalancings of institutional investors and the general price discovery mechanism). So the goal is to spot these deviances up to 15min before the close and take a long or short position in the stock. 

## Process
- The data set includes stock and future prices from 2015 to 2021 and is split into a training and test set.
- The intra-day returns (from 09:30 until 16:00) of all available stocks (US Stocks only) are regressed on all available intraday returns of the futures (using the training set data)
- High potential combinations are selected by an arbitrary adjusted-R2 threshold
- A 100-day rolling regression is performed for all the selected high potential combinations and for every potential trade entry time (i.e. 15:45, 15:50, 15:55; using the returns from open till potential trade entry time)
- Using the regression coefficients from the rolling regression, the stock return is predicted and the difference between prediction and current intra-day return is calculated
- A backtest is conducted to find the best performing combination of entry threshold (i.e. a trade is entered if the difference between predicted return and actual return is larger than the threshold), entry time, stop-loss, and take-profit
- Based on the trading performance on the training set, the best strategy per stock is selected and stocks that did not perform well are discarded
- The selected strategies per stock are aggregated to one strategy. The portfolio capital is split to entry times according to the share of stocks that use that entry time, meaning that always a fixed share of the capital is ready to be deployed at 15:45, 15:50, and 15:55. This is necessary as at 15:45 we do not know yet what opportunities might come up at 15:50 and 15:55. Unfortunately, this also causes some empty capital as we always keep capital for later entries even though there might not be one.
- Plots for the overall strategy as well for the individual stocks are generated to display the strategies performance in the training and the test set

## Findings & thoughts

