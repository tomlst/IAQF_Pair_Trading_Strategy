import os
from copula_trigger_function import CopulaTrigger
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


import scienceplots
plt.style.use(['science','no-latex'])
import statsmodels.api as sm


# return bets: Y = beta * X
def cointegration_factor(Y,X):
    return sm.OLS(Y, X).fit().params[0]

def plot_pair_and_spread(prices, start_of_present):

    historical = prices[prices.index < start_of_present]
    present = prices[prices.index >= start_of_present]

    beta = cointegration_factor(historical.iloc[:,1],historical.iloc[:,0])
    
    
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},figsize=(16,10))

    a0.set_title("Y and beta*X")
    a0.plot(historical.iloc[:,1], color='blue',linestyle='dashed')
    a0.plot(historical.iloc[:,0]*beta, color='orange',linestyle='dashed')
    a0.plot(present.iloc[:,1], color='blue')
    a0.plot(present.iloc[:,0]*beta, color='orange')

    a1.set_title("spread = Y - beta*X")
    a1.plot(historical.iloc[:,1] - historical.iloc[:,0]*beta)
    a1.plot(present.iloc[:,1] - present.iloc[:,0]*beta)

    f.tight_layout()

    return f, (a0, a1)

def generate_signals(prices,historical,threshold):
    y = prices.iloc[:,1].copy()
    x = prices.iloc[:,0].copy()

    _, result_pkg = CopulaTrigger("gumbel",[0.01,0.01],historical,threshold,plot_figure=False,recalc=True,result_package=[], spread_threshold=0)
    optimal_copula = result_pkg[13]
    print("optimal copula used: ", optimal_copula)
    _, result_pkg = CopulaTrigger(optimal_copula,[0.02,0.02],historical,threshold,plot_figure=False,recalc=True,result_package=[], spread_threshold=0)

    log_ret_x = np.log(x.pct_change()+1)
    log_ret_y = np.log(y.pct_change()+1)

    num_units = pd.Series(data=0, index=y.index, name='trade_signal')

    for i in range(1,len(y)):
        signal_text, _ = CopulaTrigger(optimal_copula, [log_ret_x[i],log_ret_y[i]],historical,threshold,plot_figure=False,recalc=False,result_package=result_pkg)

        if signal_text == "ShortYLongX": num_units[i] = -1
        elif signal_text == "ShortXLongY": num_units[i] = 1
        elif signal_text == "Hold": num_units[i] = 0
        else: num_units[i] = np.nan

        # print("in loop")
    return num_units

def generate_signals_dynamic(prices, threshold, freq, rolling=True):

    """
    this function will generate the signal sequence on a rolling basis

    prices [X,Y]: a pandas df that contains n yr prices of X and Y (will be splitted into (n-1)yr historical + 1yr present)
    freq [1,2,4,12]: define the frequency of updating the historical price
    rolling: True -> rolling basis; False -> historical + new data
    """
    prices = prices.copy()
    prices["year"] = pd.DatetimeIndex(prices.index).year
    prices["month"] = pd.DatetimeIndex(prices.index).month
    prices["quarter"] = pd.DatetimeIndex(prices.index).quarter

    # annualy
    if freq == 1:
        prices["sequence"] = prices["year"]
    # quartely
    elif freq == 4:
        prices["sequence"] = prices["year"]*10+  prices["quarter"]
    # monthly
    elif freq == 12:
        prices["sequence"] = prices["year"]*100  + prices["month"]
    else:
        raise Exception("INVALID FREQUENCY")

    periods = prices.sequence.unique()
    periods_indicies_end = np.cumsum(prices["sequence"].value_counts().sort_index())-1
    periods_indicies_end = pd.concat([pd.Series([-1]), periods_indicies_end])
    periods_indicies_end.index = range(0,len(periods_indicies_end))

    hist_start_ind = 0
    cur_start_ind = len(periods_indicies_end)-freq-1

    present_data = prices.iloc[periods_indicies_end[cur_start_ind]+1:,:]

    all_signals = pd.Series()
    while cur_start_ind < len(periods_indicies_end)-1:
        print("historical", periods_indicies_end[hist_start_ind]+1, periods_indicies_end[cur_start_ind])
        print("present", periods_indicies_end[cur_start_ind]+1, periods_indicies_end[cur_start_ind+1])

        if cur_start_ind == len(periods_indicies_end)-2:
            # print(prices.iloc[:,0][periods_indicies_end[cur_start_ind]+1: ])
            signals = generate_signals(prices.iloc[:,[0,1]][periods_indicies_end[cur_start_ind]+1: ], \
                                     prices.iloc[:,[0,1]][periods_indicies_end[hist_start_ind]+1: periods_indicies_end[cur_start_ind]], \
                                    threshold)
        else:
            signals = generate_signals(prices.iloc[:,[0,1]][periods_indicies_end[cur_start_ind]+1: periods_indicies_end[cur_start_ind+1]+1], \
                                     prices.iloc[:,[0,1]][periods_indicies_end[hist_start_ind]+1: periods_indicies_end[cur_start_ind]], \
                                    threshold)
        all_signals = all_signals.append(signals)
        
        if rolling: 
            hist_start_ind = hist_start_ind + 1
        
        cur_start_ind = cur_start_ind + 1

    return present_data,all_signals

def calculate_return(prices,beta,signals,trade_type):
    """
    Y: price of ETF Y
    X: price of ETF X
    beta: cointegration ratio Y = beta * X
    trade_type: indicate thw way to trade
    """
    
    # get copy of series
    y = prices.iloc[:,1].copy()
    y.name = 'y'
    x = prices.iloc[:,0].copy()
    x.name = 'x'

    signals.name = "signals"
    
    action_unit = pd.Series(data=0, index=y.index)
    # print(len(action_unit))
    if trade_type == "mispricing_method":
        action_unit = signals.diff()
        action_unit[0:1] = 0
        action_unit[-1] = -signals[-2]
    elif trade_type == "cumulative1":
        action_unit = signals.copy()
        action_unit[-1] = 0
        action_unit[-1] = -1 * np.sum(action_unit)
        signals[-1] = np.sign(np.sum(signals[:-1])) * -1
    elif trade_type == "cumulative2":
        # will force to clear position when changing sign
        action_unit = signals.copy()
        cum_count = 0

        action_unit[0] = signals[0]
        cum_count = signals[0]
        for i in range(1, len(action_unit)-1):
            # case: different sign -> force clear
            if cum_count * signals[i] < 0:
                action_unit[i] = -cum_count
                cum_count = 0
            else:
                action_unit[i] = signals[i]
                cum_count = cum_count + signals[i]
        action_unit[-1] = -cum_count
    action_unit.name = 'action_unit'
    position_unit = np.cumsum(action_unit)
    position_unit.name = "position_unit"

    if trade_type == "cumulative1":
        # long = 1, exit = -1
        long_short_positions = pd.Series(data=np.sign(position_unit), index=y.index, name='long_short_positions')
        
        # enter = 1, exit = -1
        enter_exit_positions = signals * long_short_positions
        enter_exit_positions[(signals != 0) & (long_short_positions == 0)] = -1
        enter_exit_positions.name = "enter_exit_positions"

        df = pd.concat([signals, action_unit, position_unit, enter_exit_positions, long_short_positions, x, y], axis=1)
        df["x_share_daily"] = 1/df["x"] * (df["action_unit"] != 0)
        df["y_share_daily"] = 1/df["y"] * (df["action_unit"] != 0)

        x_share_cum = pd.Series(data=0.0 * len(y), index=y.index, name='x_share_cum')
        y_share_cum = pd.Series(data=0.0 * len(y), index=y.index, name='y_share_cum')

        for i in range(1,len(y)):
            if df["enter_exit_positions"][i] == -1:
                x_share_cum[i] = x_share_cum[i-1] * (1 - np.abs(df["action_unit"][i] / df["position_unit"][i-1]))
                y_share_cum[i] = y_share_cum[i-1] * (1 - np.abs(df["action_unit"][i] / df["position_unit"][i-1]))
            else:
                x_share_cum[i] = x_share_cum[i-1] +  df["x_share_daily"][i]
                y_share_cum[i] = y_share_cum[i-1] +  df["y_share_daily"][i]

        df['x_share_cum'] = x_share_cum
        df['y_share_cum'] = y_share_cum

        df['unrealized_pnl_daily'] = (df["y"]*df["y_share_cum"] - df["x"]*df["x_share_cum"]) * df["long_short_positions"]
        df['realized_pnl_daily'] = (df["y"]*np.abs(df["y_share_cum"].diff()) - df["x"]*np.abs(df["x_share_cum"].diff())) * (df["enter_exit_positions"] == -1) * df["signals"]*-1

        df['realized_pnl_cum'] = np.cumsum(df['realized_pnl_daily'])
        df['unrealized_pnl_cum'] = df['realized_pnl_cum'] + df['unrealized_pnl_daily']
        df['unrealized_PnL'] = df['unrealized_pnl_cum'] - df["realized_pnl_cum"]
        
        df['x_ret_daily'] = (df['enter_exit_positions'] == -1) * (df['x'] * (np.abs(df['x_share_cum'].diff()))-1)
        df['x_ret_daily'] = df['x_ret_daily'].fillna(0)
        df['y_ret_daily'] = (df['enter_exit_positions'] == -1) * (df['y'] * (np.abs(df['y_share_cum'].diff()))-1)
        df['y_ret_daily'] = df['y_ret_daily'].fillna(0)

        df['ret_daily'] = (df['y_ret_daily'] - df['x_ret_daily']) * df['signals'] * (-1)
        df['ret_cum'] = np.cumprod(df['ret_daily']+1)

    else: 
        # enter = 1, exit = -1
        enter_exit_positions = pd.Series(data=0 * len(y), index=y.index, name='enter_exit_positions')

        for i in range(1,len(enter_exit_positions)):
            if (position_unit[i-1] == 0 and position_unit[i] != 0): enter_exit_positions[i] = 1
            elif (position_unit[i-1] != 0 and position_unit[i] == 0): enter_exit_positions[i] = -1
            else: enter_exit_positions[i] = 0

        # long = 1, exit = -1
        long_short_positions = pd.Series(data=np.sign(action_unit + position_unit), index=y.index, name='long_short_positions')

        df = pd.concat([signals, action_unit, position_unit, enter_exit_positions, long_short_positions, x, y], axis=1)
        df["x_share_daily"] = 1/df["x"] * (df["action_unit"] != 0)
        df["y_share_daily"] = 1/df["y"] * (df["action_unit"] != 0)

        x_share_cum = pd.Series(data=0.0 * len(y), index=y.index, name='x_share_cum')
        y_share_cum = pd.Series(data=0.0 * len(y), index=y.index, name='y_share_cum')

        for i in range(1,len(y)):
            if df["enter_exit_positions"][i] == 1:
                x_share_cum[i] = df["x_share_daily"][i] * np.abs(df["action_unit"][i])
                y_share_cum[i] = df["y_share_daily"][i] * np.abs(df["action_unit"][i])
            elif df["enter_exit_positions"][i] == -1:
                x_share_cum[i] = x_share_cum[i-1]
                y_share_cum[i] = y_share_cum[i-1]
            elif df["enter_exit_positions"][i-1] == -1 and df["enter_exit_positions"][i] == 0:
                x_share_cum[i] = 0
                y_share_cum[i] = 0
            else:
                x_share_cum[i] = x_share_cum[i-1] + df["x_share_daily"][i]*np.abs(df["action_unit"][i])
                y_share_cum[i] = y_share_cum[i-1] + df["y_share_daily"][i]*np.abs(df["action_unit"][i])

        df['x_share_cum'] = x_share_cum
        df['y_share_cum'] = y_share_cum

        df['unrealized_pnl_daily'] = (df["y"]*df["y_share_cum"] - df["x"]*df["x_share_cum"]) * long_short_positions
        df['realized_pnl_daily'] = df['unrealized_pnl_daily'] * df["enter_exit_positions"]
        df['realized_pnl_cum'] = np.cumsum(df['realized_pnl_daily'])
        df['unrealized_pnl_cum'] = df['realized_pnl_cum'] + df['unrealized_pnl_daily'] * (df["enter_exit_positions"] != -1)
        df['unrealized_PnL'] = df['unrealized_pnl_cum'] - df["realized_pnl_cum"]
        
        df['x_ret_daily'] = (df['enter_exit_positions'] == -1) * ((df['x'] / (np.abs(df['action_unit'])/df['x_share_cum']))-1)
        df['x_ret_daily'] = df['x_ret_daily'].fillna(0)
        df['y_ret_daily'] = (df['enter_exit_positions'] == -1) * ((df['y'] / (np.abs(df['action_unit'])/df['y_share_cum']))-1)
        df['y_ret_daily'] = df['y_ret_daily'].fillna(0)

        df['ret_daily'] = (df['y_ret_daily'] - df['x_ret_daily']) * df['long_short_positions'] * -1
        df['ret_cum'] = np.cumprod(df['ret_daily']+1)
        
    df["realized_max_drawdown"] = ((df["realized_pnl_cum"].cummax() - df["realized_pnl_cum"])/df["realized_pnl_cum"].cummax()).fillna(0).replace([np.inf,-np.inf],0)
    df["realized_avg_drawdown"] = df['realized_max_drawdown'].expanding().mean().fillna(0)

    df["unrealized_max_drawdown"] = ((df["unrealized_pnl_cum"].cummax() - df["unrealized_pnl_cum"])/df["unrealized_pnl_cum"].cummax()).fillna(0).replace([np.inf,-np.inf],0)
    df["unrealized_avg_drawdown"] = df['unrealized_max_drawdown'].expanding().mean().fillna(0)        

    return  df

def plot_summary(df):

    f, (a0, a2, a3, a4) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2, 1, 1,1]})

    f.set_figheight(10)
    f.set_figwidth(10)

    a0.set_title("P&L")
    a0.plot(df["unrealized_pnl_cum"], label="P&L")
    a0.plot(df["realized_pnl_cum"], label="Realized P&L")
    a0.plot(df["unrealized_PnL"], label="Unrealized P&L")
    a0.grid()
    a0.legend()

    '''
    a1.set_title("Realized Return %")
    a1.plot(df["ret_cum"])
    a1.grid()
    '''

    a2.set_title("Holdings")
    a2.plot(df["position_unit"])
    a2.grid()

    a3.set_title("Realized Maximum Drawdown")
    a3.plot(df["realized_max_drawdown"], label="Maximum Drawdown")
    a3.plot(df["realized_avg_drawdown"], label="Average Drawdown")
    a3.grid()

    a4.set_title("Unrealized Maximum Drawdown")
    a4.plot(df["unrealized_max_drawdown"], label="Maximum Drawdown")
    a4.plot(df["unrealized_avg_drawdown"], label="Average Drawdown")
    a4.grid()

    f.tight_layout()

    return f, (a0, a2, a3, a4)

def plot_max_drawdown(df):

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    plt.grid()
    a0.set_title("Realized Maximum Drawdown")
    a0.plot(df["realized_max_drawdown"], label="Maximum Drawdown")
    a0.plot(df["realized_avg_drawdown"], label="Average Drawdown")
    a0.legend()

    a1.set_title("Unrealized Maximum Drawdown")
    a1.plot(df["unrealized_max_drawdown"], label="Maximum Drawdown")
    a1.plot(df["unrealized_avg_drawdown"], label="Average Drawdown")
    a1.legend()


    f.tight_layout()
    return f, (a0, a1)
   

'''
import yfinance as yf
pair =  ["FDIS","XLK"]
prices_all = yf.download(pair,start="2020-01-01", end="2022-12-31")["Adj Close"]
start_of_present = "2022-01-01"

prices_present, signals_copula = generate_signals_dynamic(prices=prices_all, threshold=0.2, freq=1, rolling=True)
return_df1 = calculate_return(prices_present,0,signals_copula,"cumulative1")
plot_summary(return_df1)
# return_df1.to_excel(os.getcwd()+"\\cumulative1_testing.xlsx")

# f,_ = plot_summary(return_df1)
# f.show()

# print("finished")
# print(os.getcwd())

'''