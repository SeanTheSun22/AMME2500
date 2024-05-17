from DisplayResults import plotResults, CartAndPendulumAnimation
from StateSpaceEquations import oscillationCartAndPendulum, getParameters
from scipy.integrate import solve_ivp
import numpy as np


tStart, tEnd = 0, 100
tSpan = [tStart, tEnd]

y0 = np.array([0, 0, 20, 0])

sol = solve_ivp(oscillationCartAndPendulum, 
                tSpan, 
                y0, 
                method='RK45', 
                t_eval=np.linspace(tStart, tEnd, 1001))

plotResults(sol.t, sol.y[0])
ani = CartAndPendulumAnimation(sol.t, sol.y[0], sol.y[2], getParameters())
ani.show()
