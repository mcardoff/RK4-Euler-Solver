import numpy as np
import matplotlib.pyplot as plt
import random

MXVAL = 100000

def generalWell(xmin, xmax, steps, f):
    assert(isinstance(steps, int))
    xa = [xmin]
    Va = [MXVAL]

    h = (xmax - xmin) / steps
    for x in np.arange(xmin + h, xmax - h, h):
        xa.append(x)
        Va.append(f(x))


    xa.append(xmax)
    Va.append(MXVAL)
    return {"xs": xa, "Vs": Va}

def squareWell(xmin, xmax, steps, height):
    return generalWell(xmin,xmax,steps,lambda x: height)
