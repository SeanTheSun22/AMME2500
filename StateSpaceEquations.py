import numpy as np
import json

class StateSpaceEquations:
    def __init__(self, parameterFilename: str):

        # Read JSON file
        with open(parameterFilename, 'r') as file:
            parameters = json.load(file)

        self.dof = parameters['dof']
        self.initialConditions = parameters['initialConditions']

        # Cart Parameters
        self.M = parameters['M']
        self.k = parameters['k']
        self.c = parameters['c']
        self.A = parameters['A']
        self.w = 2 * np.pi * parameters['f']

        # Pendulum Parameters
        if self.dof > 1:
            self.m = parameters['m']
            self.L = parameters['L']
            self.g = parameters['g']
    
    def oscillation(self) -> callable:
        if self.dof == 1:
            return self.oscillationCart
        elif self.dof == 2:
            return self.oscillationCartAndPendulum
        elif self.dof == 3:
            return self.oscillationCartAndDoublePendulum

    def getParameters(self) -> dict:
        if self.dof == 1:
            return {
                'M': self.M,
                'k': self.k,
                'c': self.c,
                'A': self.A,
                'w': self.w,
                'dof': self.dof
            }
        elif self.dof > 1:
            return {
                'M': self.M,
                'k': self.k,
                'c': self.c,
                'A': self.A,
                'w': self.w,
                'm': self.m,
                'L': self.L,
                'g': self.g,
                'dof': self.dof
            }

    def forcingCart(self, A: float, w: float, t: float) -> float:
        return A * np.cos(w*t)

    def oscillationCart(self, t: float, y: list) -> list:
        # y[0] = x
        # y[1] = x_dot

        # Unpack y
        x1 = y[0]
        x2 = y[1]

        # Compute Relevant Parameters
        ft = self.forcingCart(self.A, self.w, t)

        # State Space Equations
        x1_dot = x2
        x2_dot = (ft - self.k * x1 - self.c * x2)/(self.M)

        return [x1_dot, x2_dot]

    def oscillationCartAndPendulum(self, t: float, y: list) -> list:
        # y[0] = x
        # y[1] = x_dot
        # y[2] = theta
        # y[3] = theta_dot

        # Unpack y
        x1 = y[0]
        x2 = y[1]
        th1 = y[2]
        th2 = y[3]

        # Compute Relevant Parameters
        ft = self.forcingCart(self.A, self.w, t)
        I = self.m * self.L**2 / 3
        r = self.L / 2

        # State Space Equations
        x1_dot = x2
        x2_dot = (self.m**2 * r**2 * self.g * np.sin(th1) * np.cos(th1) + I * self.m * r * th2**2 * np.sin(th1) + I * (ft - self.k * x1 - self.c * x2)) / (I * (self.M + self.m) - self.m**2 * r**2 * np.cos(th1)**2)
        
        th1_dot = th2
        th2_dot = (-(self.M + self.m) * self.m * self.g * r * np.sin(th1) - self.m * r * np.cos(th1) * (self.m * r * th1**2 * np.sin(th1) + ft - self.k * x1 - self.c * x2)) / (I * (self.M + self.m) - self.m**2 * r**2 * np.cos(th1)**2)
        
        return [x1_dot, x2_dot, th1_dot, th2_dot]
    
    def oscillationCartAndDoublePendulum(self, t: float, y: list) -> list:
        # y[0] = x
        # y[1] = x_dot
        # y[2] = theta
        # y[3] = theta_dot

        # Unpack y
        x1 = y[0]
        x2 = y[1]
        th1 = y[2]
        th2 = y[3]
        th3 = y[4]
        th4 = y[5]

        # Compute Relevant Parameters
        ft = self.forcingCart(self.A, self.w, t)
        I = self.m * self.L**2 / 12
        r = self.L / 2

        # State Space Equations
        
        A11 = self.M + 2 * self.m
        A12 = 3 * self.m * r * np.cos(th1)
        A13 = self.m * r * np.cos(th3)
        A21 = 3 * self.m * r * np.cos(th1)
        A22 = 5 * self.m * r**2 + I
        A23 = 2 * self.m * r**2 * np.cos(th1 - th3)
        A31 = self.m * r * np.cos(th3)
        A32 = 2 * self.m * r**2 * np.cos(th1 - th3)
        A33 = self.m * r**2 + I

        b1 = 3 * self.m * r * th2**2 * np.sin(th1) + self.m * r * th4**2 * np.sin(th3) - self.k * x1 - self.c * x2 + ft
        b2 = -3 * self.m * self.g * r * np.sin(th1) - 2 * self.m * r**2 * th4**2 * np.sin(th1 - th3)
        b3 = 2 * self.m * r**2 * th2**2 * np.sin(th1 - th3) - self.m * self.g * r * np.sin(th3)

        # Ax = b
        A = np.array([[A11, A12, A13],
                      [A21, A22, A23],
                      [A31, A32, A33]])
        
        b = np.array([b1, b2, b3]).transpose()

        x = np.linalg.solve(A, b)

        return [x2, x[0], th2, x[1], th4, x[2]]