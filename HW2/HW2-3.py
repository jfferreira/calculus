"""
Use Simpson's rule to compute the cumulative distribution of the standard normal
variable with 10e-12 tolerance.
Compute N(0.1), N(0.5) and N(1) with 12 digits accuracy. Start with n intervals
and double the number of intervals until the desired accuracy is achieved.
Report the approximate values you obtained for each interval until convergence,
for each of the two integrals.
"""
import numpy as np
def main():
    tol = 1e-12
    print("value: 0.1")
    t1 = cumulative_normal_distribution(0.1, tol)
    print("cumulative normal distribution: " + str(t1))
    print("value: 0.5")
    t2 = cumulative_normal_distribution(0.5, tol)
    print("cumulative normal distribution: " + str(t2))
    print("value: 1")
    t3 = cumulative_normal_distribution(1.0, tol)
    print("cumulative normal distribution: " + str(t3))
    
    
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
        integral_value = estimate_integral(t, 0, simpson_approximation, f_exp,  tol)
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
