from DisplayResults import plotResults, CartAnimation

from StateSpaceEquations import oscillationCart, getParameters
from scipy.integrate import solve_ivp
import numpy as np


tStart, tEnd = 0, 100
tSpan = [tStart, tEnd]

y0 = np.array([0, 1])

sol = solve_ivp(oscillationCart, 
                tSpan, 
                y0, 
                method='RK45', 
                t_eval=np.linspace(tStart, tEnd, 1001))

plotResults(sol.t, sol.y[0])
ani = CartAnimation(sol.t, sol.y[0], getParameters())
ani.show()
