import yfinance as yf 
import pandas as pd
import numpy as np
from scipy import stats
from copulas import *
import matplotlib.pyplot as plt
import ndtest

def fit_marginals(tickers, form_start='2017-1-1', form_end='2018-12-31', trade_start='2019-1-1', trade_end='2019-12-31'):
    #get ticker daily prices (Use Adjusted Close)
    prices = yf.download(tickers, form_start, trade_end)
    prices = pd.DataFrame(prices['Adj Close'])

    #calculate log returns
    log_returns = np.log(prices).diff().dropna()

    returns_form = log_returns.loc[form_start:form_end]
    returns_trade = log_returns.loc[trade_start:trade_end]

    #fit marginals
    marginals_df = pd.DataFrame(index=tickers, columns=['Distribution', 'AIC', 'BIC', 'KS_pvalue'])
    dist_chosen = []

    for stock in tickers:
        # print(stock)
        data = returns_form[stock]
        dists = ['Normal', "Student's t", 'Logistic', 'Extreme']
        best_aic = np.inf
        
        for dist,name in zip([stats.norm, stats.t, stats.genlogistic, stats.genextreme], dists):
            params = dist.fit(data)
            dist_fit = dist(*params)
            log_like = np.log(dist_fit.pdf(data)).sum()
            aic = 2*len(params) - 2 * log_like
            if aic<best_aic:
                best_dist = name
                best_aic = aic
                best_bic = len(params) * np.log(len(data)) - 2 * log_like
                ks_pval = stats.kstest(data, dist_fit.cdf, N=100)[1]

        marginals_df.loc[stock] = [best_dist, best_aic, best_bic, ks_pval]
        dist_chosen.append(best_dist)
        #print(marginals_df)

    return marginals_df

def fit_copulas(tickers, form_start='2014-1-2', form_end='2015-12-31', trade_start='2016-1-1', trade_end='2016-12-31'):
    #get ticker daily prices (Use Adjusted Close)
    prices = yf.download(tickers, form_start, trade_end)
    prices = pd.DataFrame(prices['Adj Close'])

    #calculate log returns
    log_returns = np.log(prices).diff().dropna()

    returns_form = log_returns.loc[form_start:form_end]
    returns_trade = log_returns.loc[trade_start:trade_end]
    
    #fit marginals
    params_ticker1 = stats.norm.fit(returns_form[tickers[0]])
    dist_t1 = stats.norm(*params_ticker1)
    params_ticker2 = stats.norm.fit(returns_form[tickers[1]])
    dist_t2 = stats.norm(*params_ticker2)

    #transform marginals 
    u = dist_t1.cdf(returns_form[tickers[0]])
    v = dist_t2.cdf(returns_form[tickers[1]])

    best_aic = np.inf
        
    for copula in [ClaytonCopula(), GumbelCopula(), FrankCopula()]:
        copula.fit(u,v)
        L = copula.log_likelihood(u,v)
        aic = 2 * copula.num_params - 2 * L
        if aic < best_aic:
            best_aic = aic
            best_copula = copula
        
    #calculate conditional probablities
    MI_Y_X = []; MI_X_Y= []
    for u,v in zip(dist_t1.cdf(returns_trade[tickers[0]]), dist_t2.cdf(returns_trade[tickers[1]])):
        MI_X_Y.append(best_copula.cdf_u_given_v(u,v))
        MI_Y_X.append(best_copula.cdf_v_given_u(u,v))

    MIXY = np.array(MI_X_Y) - 0.5; MIYX = np.array(MI_Y_X) - 0.5
    probs_trade = pd.DataFrame(np.vstack([MIXY, MIYX]).T, index = returns_trade.index, columns=['MIXY', 'MIYX'])

    return probs_trade, best_copula

def CMPI_trading(probs_trade, D, strategy):

    MIXY = probs_trade['MIXY'].to_list()
    MIYX = probs_trade['MIYX'].to_list()   
    Flag_Y = np.full(len(MIXY), np.nan)
    Flag_X = np.full_like(Flag_Y, np.nan)
    position = np.full_like(Flag_Y, np.nan)
    position[0] = 0
    Flag_Y[0] = MIYX[0]; Flag_X[0] = MIXY[0]

    flag = []; signal = ''
    if strategy == '1':
        for t in range(len(MIYX)-1):
            Flag_Y[t+1] = Flag_Y[t] + MIYX[t+1]
            Flag_X[t+1] = Flag_X[t] + MIXY[t+1]
            #print(position[t])
            if position[t] != 0:
                if np.abs(flag[t+1]) > 2 or (flag[t+1] >= 0.0 and (signal == 'longY' or signal == 'longX')) or \
                    (flag[t+1] <= 0.0 and (signal == 'shortY' or signal == 'shortX')):
                    position[t+1] = 0
                    Flag_X[t+1] = 0
                    Flag_Y[t+1] = 0
                    signal = ''
                    flag = []
                else:
                    position[t+1] = position[t]
            else:
                if Flag_Y[t+1] <= -D or Flag_X[t+1] >= D:
                    position[t+1] = 1
                    if Flag_Y[t+1] <= -D:
                        flag = Flag_Y; signal = 'longY'
                    elif Flag_X[t+1] >= D:
                        flag = Flag_X; signal = 'shortX'
                elif Flag_Y[t+1] >= D or Flag_X[t+1] <= -D:
                    position[t+1] = -1
                    if Flag_Y[t+1] >= D:
                        flag = Flag_Y; signal = 'shortY'
                    elif Flag_X[t+1] <= -D:
                        flag = Flag_X; signal = 'longX'
                else:
                    position[t+1] = 0

    elif strategy == '2':
        for t in range(len(MIYX)-1):
            Flag_Y[t+1] = Flag_Y[t] + MIYX[t+1]
            Flag_X[t+1] = Flag_X[t] + MIXY[t+1]
            #print(position[t])
            if position[t] != 0:
                if((Flag_Y[t+1] >= 0.0 or Flag_X[t+1] <= 0.0) and (signal == 'longY')) or \
                    ((Flag_Y[t+1] <= 0.0 or Flag_X[t+1]>= 0.0) and (signal == 'shortY')):
                    position[t+1] = 0
                    signal = ''
                    Flag_Y[t+1] = 0; Flag_X[t+1] = 0 
                else:
                    position[t+1] = position[t]
            else:
                if Flag_Y[t+1] <= -D and Flag_X[t+1] >= D:
                    position[t+1] = 1
                    signal = 'longY'
                elif Flag_Y[t+1] >= D and Flag_X[t+1] <= -D:
                    position[t+1] = -1
                    signal = 'shortY'
                else:
                    position[t+1] = 0
       
    positions = pd.DataFrame(position, index = probs_trade.index, columns=['position'])
    

    return positions, position


if __name__ == "__main__":
    pairs = [['FENY', 'VDE'], ['SCHE', 'VWO'], ['FENY', 'IYE']]

    thresholds = [0.4, 0.6, 0.8]
    FENY_VDE = pd.DataFrame()

    for threshold in thresholds:
        positions, position = CMPI_trading(pairs[0], threshold)
        FENY_VDE[f'D = {threshold}'] = position

    FENY_VDE = FENY_VDE.set_index(positions.index)
    FENY_VDE.to_csv('FENY_VDE_positions')
            

        