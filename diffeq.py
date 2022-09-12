import numpy as np
import matplotlib.pyplot as plt
import random
from potentials import *

def rknsolve(x0, y0, fx, fy, t0, tf, n, V, energy, iterfunc):
    """Solve a system of differential equations with eigenvalue energy"""
    h = (tf - t0) / n
    ts = V["xs"]
    Vs = V["Vs"]
    xs = [x0]
    ys = [y0]

    oldx, oldy = xs[0], ys[0]
    for i in range(1,len(ts)):
        t = ts[i]
        v = Vs[i]
            
        (newxDelta, newyDelta) = iterfunc(t,oldx,oldy,fx,fy,h,energy,v)

        newx = oldx + newxDelta
        newy = oldy + newyDelta

        oldx, oldy = newx, newy

        xs.append(oldx)
        ys.append(oldy)
        
    return (ts, xs, ys)

def rk4iter(t, x, y, fx, fy, h, E, V):
    k0, l0 = h*fx(x,y,t), h*fy(x,y,t,E,V)
    k1, l1 = h*fx(x+(k0/2), y+(l0/2), t+(h/2)), h*fy(x+(k0/2), y+(l0/2), t+(h/2),E,V)
    k2, l2 = h*fx(x+(k1/2), y+(l1/2), t+(h/2)), h*fy(x+(k1/2), y+(l1/2), t+(h/2),E,V)
    k3, l3 = h*fx(x+h, y+h, t+h), h*fy(x+h, y+h, t+h,E,V)

    return (k0 + 2*k1 + 2*k2 + k3, l0 + 2*l1 + 2*l2 + l3)

def euleriter(t, x, y, fx, fy, h, E, V):
    k0, l0 = h*fx(x,y,t), h*fy (x,y,t,E,V)

    return (k0,l0)

def eulersolve(x0, y0, fx, fy, t0, tf, n, V):
    return rknsolve(x0,y0,fx,fy,t0,tf,n,V,1.0,1.5,0.2,euleriter)

def rk4solve(x0, y0, fx, fy, t0, tf, n, V):
    final_xs = []
    final_ys = []
    final_ts = []
    first_run = True
    for energy in np.arange(1.0,10.0,0.5):
        tval,xval,yval = rknsolve(x0,y0,fx,fy,t0,tf,n,V,energy,rk4iter)
        if first_run:
            final_ts = tval
            first_run = False
        final_xs.append(xval)
        final_ys.append(yval)
    return (final_ts,final_xs,final_ys)

def eigenvalue_solve(x0,y0,fx,fy,t0,tf,n,V,estart,estop,estep):
    precision = 1e-9
    final_xs = []
    final_ys = []
    final_ts = []
    first_run = True
    prev_x, prev_energy = 0.0, 0.0
    
    for energy in np.arange(estart,estop,estep):
        tvals,xvals,yvals = rknsolve(x0,y0,fx,fy,t0,tf,n,V,energy,rk4iter)
        if first_run:
            final_ts = tvals
            first_run = False # these are set we are done

        xval = xvals[-1]
        new_xvals = []
        if np.sign(xval) != np.sign(prev_x):
            # sign change detected, use method of secants
            right, left = energy, prev_energy
            checked_val = xval
            while abs(checked_val) > precision:
                next_energy = right - checked_val * (right - left) / (checked_val - prev_x)
                _,new_xvals,_ = rknsolve(x0,y0,fx,fy,t0,tf,n,V,next_energy,rk4iter)
                left = right
                right = next_energy
                checked_val = new_xvals[-1]
            xvals.append(new_xvals)

        prev_energy = energy
        prev_x = xval

    return (final_ts,final_xs,final_ys)
