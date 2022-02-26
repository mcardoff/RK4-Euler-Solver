import numpy as np
import matplotlib.pyplot as plt
import random

#
# x(t+h) = x(t) + h*fx(x,y,t) 
# y(t+h) = y(t) + h*fy(x,y,t)
#

def eulersolve(x0, y0, fx, fy, t0, tf, n):
    """ Solve a system of differential equations as a BVP """
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n+1)
    xs = [x0]
    ys = [y0]
    oldx, oldy = xs[0], ys[0]
    
    for t in ts[1:]:
        # print(t)
        newx = oldx + h*fx(oldx,oldy,t)
        newy = oldy + h*fy(oldx,oldy,t)
        xs.append(newx)
        ys.append(newy)
        oldx, oldy = newx, newy

    return (ts, xs, ys)

def rk4solve(x0, y0, fx, fy, t0, tf, n):
    """ Solve a system of differential equations as a BVP """
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n+1)
    xs = [x0]
    ys = [y0]
    oldx, oldy = xs[0], ys[0];
    
    for t in ts[1:]:
        k0, l0 = h*fx(oldx,oldy,t), h*fy(oldx,oldy,t)
        k1, l1 = h*fx(oldx+(k0/2), oldy+(l0/2), t+(h/2)), h*fy(oldx+(k0/2), oldy+(l0/2), t+(h/2))
        k2, l2 = h*fx(oldx+(k1/2), oldy+(l1/2), t+(h/2)), h*fy(oldx+(k1/2), oldy+(l1/2), t+(h/2))
        k3, l3 = h*fx(oldx+h, oldy+h, t+h), h*fy(oldx+h, oldy+h, t+h)

        newx = oldx + (k0+2*k1+2*k2+k3)/6
        newy = oldy + (l0+2*l1+2*l2+l3)/6

        xs.append(newx)
        ys.append(newy)
        oldx, oldy = newx, newy

    return (ts, xs, ys)

def rmsErr(predicted, actual):
    assert(len(predicted) == len(actual))
    runningSum = 0.0
    for (p,a) in zip(predicted, actual):
        diff = p-a
        runningSum += diff * diff
    return np.sqrt(runningSum/len(actual))

def difference(l1, l2):
    assert(len(l1) == len(l2))
    l3 = []
    for (i1, i2) in zip(l1,l2):
        l3.append(i1-i2)

    return l3

def main():
    # insert code here
    x0, y0 = 0, 1
    fx = lambda x, y, t: y  # harmonic oscillator
    fy = lambda x, y, t: -x # harmonic oscillator
    t0, tf = 0, 10 # Time interval to solve for
    n = 10000
    ts, exs, eys = eulersolve(x0,y0,fx,fy,t0,tf,n)
    ts, rxs, rys = rk4solve(x0,y0,fx,fy,t0,tf,n)
    analytic = np.sin(ts)
    print("RMS Error RK: {}\nRMS Error Eu: {}\n".format(
        rmsErr(rxs,analytic),rmsErr(exs,analytic)))
    diffsR = max(difference(analytic, rxs))
    diffsE = max(difference(analytic, exs))
    print("Max Diff RK: {}\nMax Diff Eu: {}\n".format(
        diffsR,diffsE))
    plt.plot(ts,exs,ts,analytic,ts,rxs)
    plt.show()
    

if __name__ == "__main__":
    main()
