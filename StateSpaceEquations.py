import numpy as np


# Constants
M = 10
k = 0
c = 10
A = 0
w = 2 * np.pi / 5

m = 5
L = 5
g = 9.81

def getParameters():
    return {
        'M': M,
        'k': k,
        'c': c,
        'A': A,
        'w': w,
        'm': m,
        'L': L,
        'g': g
    }

def forcingCart(A, w, t) -> float:
    return A * np.cos(w*t)

def oscillationCartAndPendulum(t, y):
    # y[0] = x
    # y[1] = x_dot
    # y[2] = theta
    # y[3] = theta_dot

    # Unpack y
    x1 = y[0]
    x2 = y[1]
    th1 = y[2]
    th2 = y[3]

    # Equations
    ft = forcingCart(A, w, t)

    x1_dot = x2
    x2_dot = (ft + m * g * np.sin(th1) * np.cos(th1) + m * L * th2**2 * np.sin(th1) / 2 - k * x1 - c * x2)/(M + m * np.sin(th1)**2)
    th1_dot = th2
    th2_dot = -3 / (2 * L) * (x2_dot * np.cos(th1) + g * np.sin(th1))
    return [x1_dot, x2_dot, th1_dot, th2_dot]

def oscillationCart(t, y):
    # y[0] = x
    # y[1] = x_dot

    # Unpack y
    x1 = y[0]
    x2 = y[1]

    # Equations
    ft = forcingCart(A, w, t)

    x1_dot = x2
    x2_dot = (ft - k * x1 - c * x2)/(M)

    return [x1_dot, x2_dot]