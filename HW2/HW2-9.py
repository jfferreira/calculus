"""
The instantaneous rate curve r(t) is given by r(t) = 0.05/(1 + 2exp(−(1 + t)^2)
Assume that interest is compounded continuously. Compute the 6 months, 1 year, and 
18 months discount factors with six decimal digits accuracy, and compute the 2 year 
discount factor with eight decimal digits accuracy, using Simpson’s Rule.
"""
import numpy as np
def main():
    tol = 1e-6
    print("value: 0.5")
    t1 = compute_discount_factor(0.5, tol)
    print("disc(0.5)=" + str(t1))
    print("value: 1")
    t2 = compute_discount_factor(1, tol)
    print("disc(1)=" + str(t2))
    print("value: 1.5")
    t3 = compute_discount_factor(1.5, tol)
    print("disc(1.5)=" + str(t3))
    print("value: 2")
    tol = 1e-8
    t4 = compute_discount_factor(2, tol)
    print("disc(2)=" + str(t4))
   
    
def simpson_approximation(a, b, n, function):
    h = (b - a)/n
    I_simpson = function(a)/6 + function(b)/6
    for i in range(1, n):
        I_simpson = I_simpson + function(a + i*h)/3
    for i in range(1, n+1):
        I_simpson = I_simpson + 2*function(a + (i - 1/2)*h)/3
    return h*I_simpson

def instantaneous_rate(x):
    exponential = np.exp(-1*(1+x)**2)
    denominator = 1 + 2*exponential
    return 0.05/denominator

def compute_discount_factor(t, tolerance):
    integral_value = estimate_integral(0, t, simpson_approximation, 
                                       instantaneous_rate,  tolerance)
    return np.exp(-1*integral_value)
    
def print_report(n, I_new, I_old):
    print(" n: " + str(n) + 
          " approx: " + str(I_new) +
          " diff: " + str(abs(I_new - I_old)) + 
          " disc: " + str(np.exp(-1*I_new)))
    
def estimate_integral(a, b, approximation, function, tol):
    n = 4
    I_old = approximation(a, b, n, function)
    print(" n: " + str(n) + 
          " approx: " + str(I_old) +
          " disc: " + str(np.exp(-1*I_old)))
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
