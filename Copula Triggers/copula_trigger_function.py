import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import scipy.stats as stats
import math
import pandas as pd
from scipy.optimize import brentq
from scipy import stats
import random
import sys
from scipy.integrate import quad
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF


import warnings
warnings.filterwarnings("ignore")




def CopulaTrigger(family,today_return,df,threshold,plot_figure=False,precision_vector=[10000,1000],recalc=True,result_package=[],spread_threshold=0.0, disable_frank=False):

    def Given_pdf(v,family):
        if  family == 'clayton':
            def funcinner(u):
                return (alpha + 1) * ((u ** (-alpha) + v ** (-alpha) - 1) ** (-2 - 1 / alpha)) * (u ** (-alpha - 1) * v ** (-alpha - 1))
    
        elif family == 'frank':
            def funcinner(u):
                num = -alpha * (np.exp(-alpha) - 1) * (np.exp(-alpha * (u + v)))
                denom = ((np.exp(-alpha * u) - 1) * (np.exp(-alpha * v) - 1) + (np.exp(-alpha) - 1)) ** 2
                return num / denom
    
        elif family == 'gumbel':
            def funcinner(u):
                A = (-np.log(u)) ** alpha + (-np.log(v)) ** alpha
                c = np.exp(-A ** (1 / alpha))
                return c * (u * v) ** (-1) * (A ** (-2 + 2 / alpha)) * ((np.log(u) * np.log(v)) ** (alpha - 1)) * (1 + (alpha - 1) * A ** (-1 / alpha))
        return funcinner
    def CheckifRare(today_percentile,family=family,threshold=threshold):
        if (today_percentile[1] == 0 and today_percentile[0] == 0) or (today_percentile[1] == 1 and today_percentile[0] == 1):
            return "Hold"
        elif (today_percentile[1] <= 0 and today_percentile[0] > 0) or (today_percentile[0] >= 1 and today_percentile[1] < 1):
            return "ShortXLongY"
        elif (today_percentile[1] >= 1 and today_percentile[0] < 1) or (today_percentile[0] <= 0 and today_percentile[1] > 0):
            return "ShortYLongX"
        else:
            minVgivenU = quad(Given_pdf(1-today_percentile[0],family),0,threshold)[0]
            maxUgivenV = quad(Given_pdf(1-today_percentile[1],family),0,1-threshold)[0]
            maxVgivenU = quad(Given_pdf(1-today_percentile[0],family),0,1-threshold)[0]
            minUgivenV = quad(Given_pdf(1-today_percentile[1],family),0,threshold)[0]
            #print(minVgivenU,maxUgivenV)
            
            if today_percentile[1] < minVgivenU and today_percentile[0] > maxUgivenV:
                return "ShortXLongY"
            elif today_percentile[1] > maxVgivenU and today_percentile[0] < minUgivenV:
                return "ShortYLongX"
            else:
                return "Hold"
        

    
    if recalc:
        rec_today_percentile = []
        rand_upper_bound = precision_vector[0]
        resample_num = precision_vector[1]
        df["ColOneReturn"] = np.log(df.iloc[:,0]/df.iloc[:,0].shift())
        df["ColTwoReturn"] = np.log(df.iloc[:,1]/df.iloc[:,1].shift())
        
        df.dropna(inplace=True)
        global alpha
        tau, p_value = stats.kendalltau(df["ColOneReturn"], df["ColTwoReturn"])
        alpha = 2*tau/(1-tau)

        '''
        plt.scatter(df["ColOneReturn"],df["ColTwoReturn"])
        '''
        
        
        '''
        reference: https://medium.com/@financialnoob/introduction-to-copulas-part-2-9de74010ed87
        
        '''

        
        def parameter_alpha(family, tau):
            if disable_frank==False:
                if  family == 'clayton':
                    return 2 * tau / (1 - tau)
                elif family == 'frank':
                    integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
                    frank_fun = lambda alpha: ((tau - 1) / 4.0  - (quad(integrand, sys.float_info.epsilon, alpha)[0] / alpha - 1) / alpha) ** 2
                    return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x 
                elif family == 'gumbel':
                    return 1 / (1 - tau)
            else: 
                if  family == 'clayton':
                    return 2 * tau / (1 - tau)
                elif family == 'gumbel':
                    return 1 / (1 - tau)
        
        #GUMBEL COPULA
    
        tau, p_value = stats.kendalltau(df["ColOneReturn"], df["ColTwoReturn"])
        alpha = parameter_alpha(family, tau)
        
        

        
        if plot_figure:
            x = np.linspace(0.99999999,0.0001, 1000)
    
            rec = []
            rec2 = []

            for i in x:
                rec.append(quad(Given_pdf(i,family),0,threshold)[0])
                rec2.append(quad(Given_pdf(i,family),0,1-threshold)[0])

            plt.figure(figsize=(10,10))
            plt.plot(np.linspace(0.0001,0.99999999,1000),rec,c="r",linestyle='dashed')
            plt.plot(rec2,np.linspace(0.0001,0.99999999,1000),c="r",linestyle='dashed')
            plt.plot(np.linspace(0.0001,0.99999999,1000),rec2,c="g",linestyle='dashed')
            plt.plot(rec,np.linspace(0.0001,0.99999999,1000),c="g",linestyle='dashed')
            #print(rec)
            #print(rec,alpha)
            today_percentile = [0.01,0.9]
            plt.scatter(today_percentile[0],today_percentile[1],c="r",s=3)
    
            #print(CheckifRare(family,threshold,today_percentile))
        
        
        
        
        def lnpdf_copula(family, alpha, u, v):
            '''Estimate the log probability density function of three kinds of Archimedean copulas
            '''
        
            if  family == 'clayton':
                pdf = (alpha + 1) * ((u ** (-alpha) + v ** (-alpha) - 1) ** (-2 - 1 / alpha)) * (u ** (-alpha - 1) * v ** (-alpha - 1))
        
            elif family == 'frank':
                #print(alpha)
                num = -alpha * (np.exp(-alpha) - 1) * (np.exp(-alpha * (u + v)))
                denom = ((np.exp(-alpha * u) - 1) * (np.exp(-alpha * v) - 1) + (np.exp(-alpha) - 1)) ** 2
                pdf = num / denom
                #num = -alpha * (np.exp(-alpha) - 1) * (np.exp(-alpha * (u + v)))
                #denom = ((np.exp(-alpha * u) - 1) * (np.exp(-alpha * v) - 1) + (np.exp(-alpha) - 1)) ** 2
                #pdf = (-alpha * (np.exp(-alpha) - 1) * (np.exp(-alpha * (u + v)))) / (((np.exp(-alpha * u) - 1) * (np.exp(-alpha * v) - 1) + (np.exp(-alpha) - 1)) ** 2)
        
            elif family == 'gumbel':
                A = (-np.log(u)) ** alpha + (-np.log(v)) ** alpha
                c = np.exp(-A ** (1 / alpha))
                pdf = c * (u * v) ** (-1) * (A ** (-2 + 2 / alpha)) * ((np.log(u) * np.log(v)) ** (alpha - 1)) * (1 + (alpha - 1) * A ** (-1 / alpha))

            return np.log(pdf)
        
        
        
        AIC ={}  
        x=df.iloc[:,2]
        y=df.iloc[:,3]
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)
        u, v = [ecdf_x(a) for a in x], [ecdf_y(a) for a in y]
        
        if disable_frank == False:
            for i in ['clayton', 'frank', 'gumbel']:
                param = parameter_alpha(i, tau)
                lnpdf = [lnpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
                lnpdf = np.nan_to_num(lnpdf) 
                loglikelihood = sum(lnpdf)
                AIC[i] = [param, -2 * loglikelihood + 2]
        else:
            for i in ['clayton', 'gumbel']:
                param = parameter_alpha(i, tau)
                lnpdf = [lnpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
                lnpdf = np.nan_to_num(lnpdf) 
                loglikelihood = sum(lnpdf)
                AIC[i] = [param, -2 * loglikelihood + 2]
        
        global copula
        copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        print(f"{copula} is optimal for {df.columns[0]} and {df.columns[1]}")



        
        if family == "gumbel":
            def gumbel_phi(t, alpha):
                return (-np.log(t))**alpha
            
            def gumbel_phi_inv(t, alpha):
                return np.exp(-t**(1/alpha))
            
            def gumbel_K(t, alpha):
                return t * (alpha - np.log(t)) / alpha
              
            t1 = np.random.rand(rand_upper_bound)
            t2 = np.random.rand(rand_upper_bound)
            
            w = []
            for t in t2:
                func = lambda w: gumbel_K(w, alpha=alpha) - t
                w.append(brentq(func, 0.0000000001, 0.9999999999))
            w = np.array(w).flatten()
            
            u = gumbel_phi_inv(t1 * gumbel_phi(w, alpha=alpha), alpha=alpha)
            v = gumbel_phi_inv((1-t1) * gumbel_phi(w, alpha=alpha), alpha=alpha)
            

        elif family == "clayton":
            def clayton_phi(t, alpha):
                return 1/alpha * (t**(-alpha) - 1)
            
            def clayton_phi_inv(t, alpha):
                return (alpha * t + 1)**(-1/alpha)
            
            def clayton_K(t, alpha):
                return t * (alpha - t**alpha + 1) / alpha
              
            t1 = np.random.rand(rand_upper_bound)
            t2 = np.random.rand(rand_upper_bound)
            
            w = []
            for t in t2:
                func = lambda w: clayton_K(w, alpha=alpha) - t
                w.append(brentq(func, 0.0000000001, 0.9999999999))
            w = np.array(w).flatten()
            
            u = clayton_phi_inv(t1 * clayton_phi(w, alpha=alpha), alpha=alpha)
            v = clayton_phi_inv((1-t1) * clayton_phi(w, alpha=alpha), alpha=alpha)
        
        elif family == "frank":
            def frank_phi(t, alpha):
                return -np.log((np.exp(-alpha*t) - 1) / (np.exp(-alpha) - 1))
            
            def frank_phi_inv(t, alpha):
                return -1/alpha * np.log((np.exp(-alpha) - 1) / np.exp(t) + 1)
            
            def frank_K(t, alpha):
                return (t + (1 - np.exp(alpha*t)) * np.log((1-np.exp(alpha*t)) * 
                                                           np.exp(-alpha*t+alpha) / (1-np.exp(alpha))) / alpha)

            t1 = np.random.rand(rand_upper_bound)
            t2 = np.random.rand(rand_upper_bound)
            
            w = []
            for t in t2:
                func = lambda w: frank_K(w, alpha=alpha) - t
                w.append(brentq(func, 0.0000000001, 0.9999999999))
            w = np.array(w).flatten()

            
            u = frank_phi_inv(t1 * frank_phi(w, alpha=alpha), alpha=alpha)
            v = frank_phi_inv((1-t1) * frank_phi(w, alpha=alpha), alpha=alpha)
            delInf = pd.DataFrame([u,v]).T.dropna()

            delInf = delInf.replace([np.inf, -np.inf], np.nan)

            delInf = delInf.dropna()
            u = delInf.iloc[:,0].to_numpy()
            v = delInf.iloc[:,1].to_numpy()

        
        picked = (np.floor(np.random.uniform(0,len(u),resample_num))).astype(int)
        picked_vec = (u[picked],v[picked])
        #plt.scatter(picked_vec[0],picked_vec[1],s=0.1)
        ret_vec = pd.DataFrame((df["ColOneReturn"].quantile(picked_vec[0]).to_numpy(),df["ColTwoReturn"].quantile(picked_vec[1]).to_numpy()))
        ret_vec = ret_vec.T
        #print(ret_vec)

        
        pairs = pd.DataFrame([u,v],index=["u","v"]).T
        if plot_figure:
            plt.scatter(pairs.iloc[:,0],pairs.iloc[:,1],s=0.1)
            #plt.scatter(pairs.iloc[:,0],pairs.iloc[:,1])
            plt.show()
        
        

        n=100
        
        Upper_Bound_of_V_Given_U = [0]
        Lower_Bound_of_V_Given_U = [0]
        Upper_Bound_of_U_Given_V = [0]
        Lower_Bound_of_U_Given_V = [0]

        #print(pairs)
        #Find 99 ,1, 40, 60 percentile in Gumbel Copula
        for i in np.linspace(1/n, 1,n-1):
            Upper_Bound_of_V_Given_U.append(pairs.iloc[:,1][pairs.iloc[:,0]<i][pairs.iloc[:,0]>(i-1/n)].quantile(1-threshold)) # Upper Bound of V given U
            Upper_Bound_of_U_Given_V.append(pairs.iloc[:,0][pairs.iloc[:,1]<i][pairs.iloc[:,1]>(i-1/n)].quantile(1-threshold)) # Lower Bound of V given U
        for i in np.linspace(1/n, 1,n-1):
            Lower_Bound_of_V_Given_U.append(pairs.iloc[:,1][pairs.iloc[:,0]<i][pairs.iloc[:,0]>(i-1/n)].quantile(threshold)) # Upper Bound of U given V
            Lower_Bound_of_U_Given_V.append(pairs.iloc[:,0][pairs.iloc[:,1]<i][pairs.iloc[:,1]>(i-1/n)].quantile(threshold)) # Lower Bound of U given V
        
        Upper_Bound_of_V_Given_U.append(1)
        Lower_Bound_of_V_Given_U.append(1)
        Upper_Bound_of_U_Given_V.append(1)
        Lower_Bound_of_U_Given_V.append(1)
        #plt_upper_bound_exit.append(1)
        #plt_lower_bound_exit.append(1)

        
        
    elif recalc == False and result_package == []:
        raise Exception("Error: You need to recalculate the framework or input previous result in the function.")
    elif recalc == False and result_package[7] != threshold:
        raise Exception("Error: You need to make sure the threshold when initialization is the same as when recalc == False")
    elif recalc == False and result_package[8] != family:
        raise Exception("Error: You need to make sure the Copula Family when initialization is the same as when recalc == False")
    else:
        df = result_package[0]
        Upper_Bound_of_V_Given_U = result_package[1]
        Lower_Bound_of_V_Given_U = result_package[2]
        Upper_Bound_of_U_Given_V = result_package[3]
        Lower_Bound_of_U_Given_V = result_package[4]
        #plt_upper_bound_exit = result_package[3]
        #plt_lower_bound_exit = result_package[4]
        picked_vec = result_package[5]
        ret_vec = result_package[6]
        rec_today_percentile = result_package[14]
    
    #print(picked_vec)
    def OutOfBound(i):

        return CheckifRare([picked_vec[0][i],picked_vec[1][i]])

        #return picked_vec[1][i] > short_y_long_x[np.floor(picked_vec[0][i]*101).astype(int)] or picked_vec[1][i] < short_x_long_y[np.floor(picked_vec[0][i]*101).astype(int)]
    #def InBound(i):
        #return picked_vec[1][i] < plt_upper_bound_exit[np.floor(picked_vec[0][i]*101).astype(int)] and picked_vec[1][i] > plt_lower_bound_exit[np.floor(picked_vec[0][i]*101).astype(int)]

    def plotfigure():
        out_indices_1 = []
        out_indices_2 = []
        resample_num = precision_vector[1]

        global spreadupper,spreadlower
        spreadupper = spread_threshold
        spreadlower = -spread_threshold
        
        for i in range(resample_num):
            print(i)
            diff = ret_vec[0][i] - ret_vec[1][i]
            if diff < spreadlower or diff > spreadupper:
                if OutOfBound(i) == "ShortYLongX":
                    out_indices_1.append(i)
                if OutOfBound(i) == "ShortXLongY":
                    out_indices_2.append(i)
        plt.figure(dpi=200)
        plt.scatter(ret_vec.iloc[:,0],ret_vec.iloc[:,1],s=0.5)
        plt.scatter(ret_vec.iloc[out_indices_1,:].iloc[:,0],ret_vec.iloc[out_indices_1,:].iloc[:,1],s=0.5,c="r")
        plt.scatter(ret_vec.iloc[out_indices_2,:].iloc[:,0],ret_vec.iloc[out_indices_2,:].iloc[:,1],s=0.5,c="purple")
        #plt.scatter(ret_vec.iloc[in_indices,:].iloc[:,0],ret_vec.iloc[in_indices,:].iloc[:,1],s=0.5,c="g")
        plt.scatter(today_return[0],today_return[1],c="black",s=2,marker=(5,1))            
        plt.title(f"{family} Copula")
        plt.grid()
        plt.show()    
        
    
    def Signal(today_return,historical_df):
        #print(historical_df)
        today_percentile = []
        '''
        diff = today_return[0] - today_return[1]
        spreadupper = spread_threshold
        spreadlower = -spread_threshold

        if diff > spreadlower and diff < spreadupper:
            return "Hold"
        '''

        if recalc:
            if plot_figure:        
                plotfigure()
            global maxx,miny,minx,maxy
            maxx = max(df.iloc[:,2])
            maxy = max(df.iloc[:,3])
            minx = min(df.iloc[:,2])
            miny = min(df.iloc[:,3])
            if today_return[0] > maxx and today_return[1] < miny:
                return "ShortXLongY"
            elif today_return[0] < minx and today_return[1] > maxy:
                return "ShortYLongX"
            elif today_return[0] > maxx or  today_return[1] < miny or \
                today_return[0] < minx or today_return[1] > maxy:
                return "Hold"

        today_percentile.append(stats.percentileofscore(historical_df.iloc[:,2], today_return[0]))
        today_percentile.append(stats.percentileofscore(historical_df.iloc[:,3], today_return[1]))
        #print(historical_df)


        today_percentile = [today_percentile[0]/100,today_percentile[1]/100]
        #print(f"Today_percentile: {today_percentile}")
        rec_today_percentile.append(today_percentile)
        if CheckifRare(today_percentile=today_percentile)=="ShortYLongX":
            return "ShortYLongX"
        elif CheckifRare(today_percentile=today_percentile)=="ShortXLongY":
            return "ShortXLongY"
        else:
            return "Hold"
            
    #print(df)
    return Signal(today_return,df),[df,Upper_Bound_of_V_Given_U, Lower_Bound_of_V_Given_U, Upper_Bound_of_U_Given_V, Lower_Bound_of_U_Given_V,picked_vec,picked_vec,threshold,family,maxx,miny,minx,maxy,copula,rec_today_percentile]


#Function Demo
'''
df = yf.download(["PEP","KO"],start="2018-01-01", end="2019-12-31")

df = df["Close"]
result_package = []

copulathrd = 0.35

today_signal, result_package =CopulaTrigger("frank",[0.02,0.02],df,copulathrd,plot_figure=True,recalc=True,result_package=result_package)
rec = []


for i in range(100):
    print(i)
    rand = random.random()
    today_signal, result_package =CopulaTrigger("frank",[df.iloc[np.floor(rand*len(df)).astype(int),2],df.iloc[np.floor(rand*len(df)).astype(int),3]],df,copulathrd,plot_figure=False,recalc=False,result_package=result_package,spread_threshold=0.0)
    print([df.iloc[np.floor(rand*len(df)).astype(int),2],df.iloc[np.floor(rand*len(df)).astype(int),3]],today_signal)
    rec.append(today_signal)


    
rec=pd.DataFrame(rec)
print(rec.value_counts())
'''