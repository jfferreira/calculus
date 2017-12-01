"""
Appendix D-2
===============================================
Bid and ask prices of the fifteen most liquid S&P 500 options on 11/21/2011 with maturity
less than one year can be found in the SPX OptionPrices 11.21.2011 file.
(i) Compute the implied volatilities corresponding to the mid-price quotes (average of Bid
and Ask quotes) of each option.
"""

import numpy as np
import datetime as dt
import pandas as pd
import statsmodels.api as sm

def main():
    # tolerance for all calculations
    tol = 1e-6
    # Parameters given
    S = 1193
    today = dt.date(2011,11,21)
    march_expiry = dt.date(2012,3,30)
    june_expiry = dt.date(2012,6,16)
    sep_expiry = dt.date(2012,9,22)
    
    t_march = np.busday_count(today, march_expiry)/252
    t_june = np.busday_count(today, june_expiry)/252
    t_sep = np.busday_count(today, sep_expiry)/252
    # option price data
    filename = 'SPX_Options_11.21.2011.csv'
    data = pd.read_csv(filename)
    # calculate mids
    data["MID"] = mid(data["PX_BID"], data["PX_ASK"])
    # add option type
    data["TYPE"] = ["C", "C", "P", "P", "P", "C", "C", "P", "P", "P", "C", "C", "P", "P", "P"]
    data["T"] = [t_march, t_march, t_march, t_march, t_march, 
                 t_june, t_june, t_june, t_june, t_june, 
                 t_sep, t_sep, t_sep, t_sep, t_sep]
    # estimate r and q using the OLS method described in "A Linear Algebra Primer for Financial 
    # Engineering", Ch. 8 using the put and call option prices for September
    A = [-1175, -1200]
    y = [129.1-127.15, 115.05-137.9]
    A = sm.add_constant(A)
    regression = sm.OLS(y, A)
    estimation = regression.fit()
    pv_fwd = estimation.params[0]
    discount = estimation.params[1]
    r = -np.log(discount)/t_sep
    q = -np.log(pv_fwd/S)/t_sep
    print("==============================")
    print ("r estimate= " + str(r))
    print ("q estimate= " + str(q))

    for row in data.itertuples():
        vol = implied_vol(S, row.OPT_STRIKE_PX, row.T, r, q, row.MID, tol, row.TYPE)
        print("==============================")
        print(row._1)
        print(vol)
    
def mid(bid, ask):
    return (bid + ask)/2
     
def implied_vol(S, K, T, r, q, price, tol, option_type="C"):
    x0 = 0.5
    x_new = x0
    x_old = x0 - 1
    while np.abs(x_new - x_old) > tol:
        x_old = x_new
        if (option_type == "C"):
            x_new = x_new - (call(S, K, T, r, q, x_new) - price)/vega(S, K, T, r, q, x_new)
        else:
            x_new = x_new - (put(S, K, T, r, q, x_new) - price)/vega(S, K, T, r, q, x_new)
    return x_new
    
def call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*np.exp(-q*T)*taylor_cum_normal_dist(d1) - K*np.exp(-r*T)*taylor_cum_normal_dist(d2)
    return call

def put(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = K*np.exp(-r*T)*taylor_cum_normal_dist(-d2) - S*np.exp(-q*T)*taylor_cum_normal_dist(-d1)
    return put

def vega(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T)/(sigma*np.sqrt(T))
    return (1/np.sqrt(2*np.pi))*S*np.exp(-q*T)*np.sqrt(T)*np.exp(-d1**2/2)

def taylor_cum_normal_dist(t):
    z = np.abs(t)
    y = 1/(1 + 0.2316419*z)
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    m = 1 - np.exp(-t**2/2)*(a1*y + a2*y**2 + a3*y**3 + a4*y**4 + a5*y**5)/np.sqrt(2*np.pi)
    if t > 0:
        return m
    else:
        return 1 - m

if __name__ == '__main__':
    main()
