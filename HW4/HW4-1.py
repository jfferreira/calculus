"""
Appendix A
===============================================
 Consider a three months ATM call with strike 40 on an underlying asset with spot price 40 
 following a lognormal distribution with volatility 20% and paying dividends continuously 
 at 1%. Assume the risk–free interest rate is constant at 5%. 
 (i) Compute the Black–Scholes value of the call using the routine from Table 3.1 for 
 computing approximate values for cumulative distributions of the standard normal variable; 
 (ii) Compute the Black–Scholes value of the call using Simpson’s rule with tolerance 10−12 
 to compute approximate values for cumulative distributions of the standard normal variable.
"""

import numpy as np
def main():
    K = 40
    S = 40
    T = 0.25
    sigma = 0.2
    q = 0.01
    r = 0.05
    
    print ("Using the routine in Table 3.1:")
    c0 = call_option_value(S, K, T, r, q, sigma, cum_dist_method="taylor")
    print("%.12f" % c0)
    print ("Using a simpson approximation:")
    c1 = call_option_value(S, K, T, r, q, sigma, cum_dist_method="simpson", tol=1e-12)
    print("%.12f" % c1)

def  call_option_value(S, K, T, r, q, sigma, cum_dist_method="simpson", tol=1e-12):
    d1 = (np.log(S/K) + (r-q+sigma**2/2)*T)/(sigma*np.sqrt(T))
    print("d1=" + str(d1))
    d2 = d1 - sigma*np.sqrt(T)
    print("d2=" + str(d1))
    if cum_dist_method.lower() == "simpson":
        return S*np.exp(-q*T)*cumulative_normal_distribution(d1, tol) 
        - K*np.exp(-r*T)*cumulative_normal_distribution(d2,tol)
    elif cum_dist_method.lower() == "taylor":
        return S*np.exp(-q*T)*taylor_cum_normal_dist(d1) 
        - K*np.exp(-r*T)*taylor_cum_normal_dist(d2)    
        

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
    
def simpson_approximation(a, b, n, function):
    h = (b - a)/n
    I_simpson = function(a)/6 + function(b)/6
    for i in range(1, n):
        I_simpson = I_simpson + function(a + i*h)/3
    for i in range(1, n+1):
        I_simpson = I_simpson + 2*function(a + (i - 1/2)*h)/3
    return h*I_simpson

def f_exp(x):
    return np.exp(-x**2/2)
    
def cumulative_normal_distribution(t, tol):
    if(t > 0):
        integral_value = estimate_integral(0, t, simpson_approximation, f_exp,  tol)
    else:
        integral_value = estimate_integral(-t, 0, simpson_approximation, f_exp,  tol)
    return 1/2 + (1/np.sqrt(2*np.pi))*integral_value

def print_report(n, I_new, I_old):
    print(" n: " + str(n) + 
          " approx: " + str(I_new) +
          " diff: " + str(abs(I_new - I_old)))
    
def estimate_integral(a, b, approximation, function, tol):
    n = 4
    I_old = approximation(a, b, n, function)
    print(" n: " + str(n) + 
          " approx: " + str(I_old))
    n = 2*n
    I_new = approximation(a, b, n, function)
    print_report(n, I_new, I_old)
    while(abs(I_new - I_old) > tol):
        I_old = I_new
        n = 2*n
        I_new = approximation(a, b, n, function)
        print_report(n, I_new, I_old)
    return I_new

if __name__ == '__main__':
    main()
