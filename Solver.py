from DisplayResults import plotResults, displayResults

from StateSpaceEquations import oscillationCart, getCartParameters
from scipy.integrate import solve_ivp
import numpy as np

def solveCart(tStart, tEnd):
    tSpan = [tStart, tEnd]

    y0 = np.array([0, 0])

    sol = solve_ivp(oscillationCart, 
                    tSpan, 
                    y0, 
                    method='RK45', 
                    t_eval=np.linspace(tStart, tEnd, 1001))

    return sol


tStart, tEnd = 0, 100
tSpan = [tStart, tEnd]

y0 = np.array([0, 0, 20, 0])

sol = solve_ivp(oscillationCart, 
                tSpan, 
                y0, 
                method='RK45', 
                t_eval=np.linspace(tStart, tEnd, 1001))

plotResults(sol.t, sol.y[0])
displayResults(sol.t, sol.y[0], sol.y[2], getCartParameters())
