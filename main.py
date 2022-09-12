import numpy as np
import matplotlib.pyplot as plt
import random
from potentials import *

from diffeq import *

#
# x(t+h) = x(t) + h*fx(x,y,t) 
# y(t+h) = y(t) + h*fy(x,y,t)
#       
def main():
    # insert code here
    x0, y0 = 0, 1
    fx = lambda psi, psip, x: psip  # Schrodinger Equation
    fy = lambda psi, psip, x, E, V: -2.0 * (E-V) * psi # Schrodinger Equation
    t0, tf = 0, 1 # x interval to solve for
    n = 1000
    # ts, rxs, rys = rk4solve(x0,y0,fx,fy,t0,tf,n,squareWell(t0,tf,n,0.0))
    ts, rxs, _ = eigenvalue_solve(x0, y0, fx, fy, t0, tf, n, squareWell(t0, tf, n, 0.0), 1.0, 10.0, 0.5)
    # analytic = np.sin(ts)

    for rx in rxs:
        plt.plot(ts,rx)
    plt.show()
    

if __name__ == "__main__":
    main()
