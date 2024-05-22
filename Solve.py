import os
from Display import plotResults, CartAnimation

from StateSpaceEquations import StateSpaceEquations
from scipy.integrate import solve_ivp
import numpy as np

tStart, tEnd = 0, 100
tSpan = [tStart, tEnd]
filename = "PendulumSwinging"

stateSpace = StateSpaceEquations(os.path.join("SavedRuns", filename) + ".json")

sol = solve_ivp(stateSpace.oscillation(), 
                tSpan, 
                stateSpace.initialConditions, 
                method='RK45', 
                t_eval=np.linspace(tStart, tEnd, 1001))

plotResults(sol.t, sol.y[0])

if stateSpace.dof == 1:
    ani = CartAnimation(sol.t, sol.y[0], stateSpace.getParameters())
if stateSpace.dof == 2:
    from Display import CartAndPendulumAnimation
    ani = CartAndPendulumAnimation(sol.t, sol.y[0], sol.y[2], stateSpace.getParameters())
ani.show()

ani.save(os.path.join("SavedVideos", filename) + ".mp4")