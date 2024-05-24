import os
from Display import ResultsPlotter, CartAnimation, CartAndPendulumAnimation

from StateSpaceEquations import StateSpaceEquations
from scipy.integrate import solve_ivp
import numpy as np

tStart, tEnd = 0, 100
tSpan = [tStart, tEnd]

# CONFIG FILE WITH ALL PARAMETERS NO NEED TO CHANGE ANYTHING ELSE EXCEPT TEND
# CLOSE THE FIGURE POP UPS TO START MODEL AND AFTER CLOSING ANIMATION IT WILL 
# SAVE TO SAVEDVIDEOS
filename = "ResonantLargeDoublePendulum"

stateSpace = StateSpaceEquations(os.path.join("SavedRuns", filename) + ".json")

sol = solve_ivp(stateSpace.oscillation(), 
                tSpan, 
                stateSpace.initialConditions, 
                method='RK45', 
                t_eval=np.linspace(tStart, tEnd, 1001))

ResultsPlotter.plot3dofValues(sol)
ResultsPlotter.plotEnergy(sol, stateSpace.getParameters())

if stateSpace.dof == 1:
    ani = CartAnimation(sol.t, sol.y[0], stateSpace.getParameters())
if stateSpace.dof == 2:
    ani = CartAndPendulumAnimation(stateSpace.getParameters(), sol.t, sol.y[0], sol.y[2])
if stateSpace.dof == 3:
    ani = CartAndPendulumAnimation(stateSpace.getParameters(), sol.t, sol.y[0], sol.y[2], sol.y[4])

ani.show()

ani.save(os.path.join("SavedVideos", filename) + ".mp4")