"""
Appendix B
===============================================
A six months at–the–money call on an underlying asset with spot price 30 paying dividends
continuously at a 1% rate is worth $2.5. Assume that the risk free interest rate is constant
at 3%. Use Newton’s method with initial guess 0.5 to compute the corresponding implied
volatility with six decimal digits accuracy.
"""

import numpy as np

def main():
    K = 30
    S = 30
    T = 0.5
    q = 0.01
    r = 0.03
    C = 2.5
    tol = 1e-6
    # initial volatility estimate
    x0 = 0.5
    x_new = x0
    x_old = x0 - 1
    print("Implied volatility - Newton's method")
    while np.abs(x_new - x_old) > tol:
        x_old = x_new
        x_new = x_new - (call(S, K, T, r, q, x_new) - C)/vega(S, K, T, r, q, x_new)
        print("x_new = %.12f"%x_new)
        print("|x_new - x_old| = %.12f"%np.abs(x_new - x_old))
        print("------------------------------------")
    
def call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*taylor_cum_normal_dist(d1) - K*np.exp(-r*T)*taylor_cum_normal_dist(d2)


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

