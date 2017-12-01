"""
Appendix C
===============================================
A five months at–the–money call on an underlying asset with spot price 40 
paying dividends continuously at a 1% rate is worth $2.75. 
Assume that the risk free interest rate is constant at 2.5%.
(i) Compute the implied volatility with six decimal digits accuracy, 
using the bisection method on the interval [0.0001, 1], 
the secant method with initial guess 0.5, 
and Newton’s method with initial guess 0.5.
"""

import numpy as np

def main():
    K = 40
    S = 40
    T = 5/12
    q = 0.01
    r = 0.025
    C = 2.75
    tol = 1e-6
    
    print("Implied volatility - Newton's method")
    print("====================================")
    x0 = 0.5
    x_new = x0
    x_old = x0 - 1
    while np.abs(x_new - x_old) > tol:
        x_old = x_new
        x_new = x_new - (call(S, K, T, r, q, x_new) - C)/call_vega(S, K, T, r, q, x_new)
        print("x_new = %.12f"%x_new)
        print("|x_new - x_old| = %.12f"%np.abs(x_new - x_old))
        print("------------------------------------")
    
    print("Implied volatility - Bisection method")
    print("====================================")
    x_left = 0.0001
    x_right = 1
    f_x_left = call(S, K, T, r, q, x_left) - C
    f_x_right = call(S, K, T, r, q, x_right) - C
    while np.maximum(np.abs(f_x_left), np.abs(f_x_right)) > tol or x_right - x_left > tol:
        f_x_left = call(S, K, T, r, q, x_left) - C
        f_x_right = call(S, K, T, r, q, x_right) - C
        x_middle = (x_left + x_right)/2
        f_x_middle = call(S, K, T, r, q, x_middle) - C
        print("x_new = %.12f"%x_middle)
        print("tol_f = %.12f"%np.maximum(np.abs(f_x_left), np.abs(f_x_right)))
        print("tol_x = %.12f"% (x_right - x_left))
        if (f_x_left*f_x_middle < 0):
            x_right = x_middle
        else:
            x_left = x_middle
        print("------------------------------------")
        
    print("Implied volatility - Secant method")
    print("====================================")
    x_new = 0.5
    x_old = x_new - 1
    f_x_new = call(S, K, T, r, q, x_new) - C
    while np.abs(f_x_new) > tol or np.abs(x_new - x_old) > tol:
        x_oldest = x_old
        x_old = x_new
        f_x_oldest = call(S, K, T, r, q, x_oldest) - C
        f_x_old = call(S, K, T, r, q, x_old) - C
        x_new = x_old - f_x_old*(x_old - x_oldest)/(f_x_old - f_x_oldest) 
        f_x_new = call(S, K, T, r, q, x_new) - C
        print("x_new = %.12f"%x_new)
        print("tol_f = %.12f"%np.abs(f_x_new))
        print("tol_x = %.12f"%np.abs(x_new - x_old))
        print("------------------------------------")
    
def call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*taylor_cum_normal_dist(d1) - K*np.exp(-r*T)*taylor_cum_normal_dist(d2)


def call_vega(S, K, T, r, q, sigma):
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

