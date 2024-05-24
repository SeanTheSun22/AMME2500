from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


class CartAnimation(animation.TimedAnimation):
    def __init__(self, t: list, x: list, cartParameters: dict):
        self.t = t
        self.x = x

        self.cartWidth = 2
        self.cartHeight = 1

        self.screenWidthBuffer = 2
        self.screenHeightBuffer = 2

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-(max(abs(self.x)) + self.screenWidthBuffer + self.cartWidth // 2), max(abs(self.x)) + self.screenWidthBuffer + self.cartWidth // 2)
        self.ax.set_ylim(-self.screenHeightBuffer, self.cartHeight + self.screenHeightBuffer)

        self.ax.set_aspect('equal', adjustable='box')

        self.timeText = self.ax.text(0.02, 0.02, f"Time: {self.t[0]:.2f}", transform=self.ax.transAxes)
        self.cart = plt.Rectangle((self.x[0] - self.cartWidth // 2, 0), self.cartWidth, self.cartHeight, fill=True, color='blue')

        self.ax.add_patch(self.cart)

        animation.TimedAnimation.__init__(self, self.fig, interval=(t[1] - t[0]) * 1000, blit=True)

    def _draw_frame(self, i: int) -> None:
        self._draw_time(i)
        self._draw_cart(i)

    def _draw_time(self, i: int) -> None:
        self.timeText.set_text(f"Time: {self.t[i]:.2f}")
    
    def _draw_cart(self, i: int) -> None:
        self.cart.set_xy([self.x[i] - self.cartWidth // 2, 0])

    def new_frame_seq(self) -> iter:
        return iter(range(len(self.t)))

    def show(self) -> None:
        plt.show()

    def save(self, filename: str) -> None:
        super().save(filename, writer='ffmpeg')
        

class CartAndPendulumAnimation(CartAnimation):
    def __init__(self, cartParameters: dict, t: list, x: list, theta1: list, theta2: list=None) -> None:
        super().__init__(t, x, cartParameters)
        
        self.theta = [theta1, theta2]
        self.pendulumLength = cartParameters["L"]
        self.dof = cartParameters["dof"]
        
        self.ax.set_ylim(-self.pendulumLength - self.screenHeightBuffer, self.cartHeight + self.screenHeightBuffer)

        self.ax.set_aspect('equal', adjustable='box')
        
        self.pendulums = []
        for i in range(self.dof - 1):
            pendulum, = self.ax.plot([], [], color='black', lw=2)
            self.pendulums.append(pendulum)        

    def _draw_frame(self, i: int) -> None:
        super()._draw_frame(i)
        self._draw_pendulum(i)

    def _draw_pendulum(self, i: int) -> None:
        pendulumX1 = self.x[i]
        pendulumY1 = 0
        for j, pen in enumerate(self.pendulums):
            pendulumX2 = pendulumX1 + self.pendulumLength * np.sin(self.theta[j][i])
            pendulumY2 = pendulumY1 + self.pendulumLength * np.cos(self.theta[j][i])
            pen.set_data([pendulumX1, pendulumX2], [-pendulumY1, -pendulumY2])
            pendulumX1 = pendulumX2
            pendulumY1 = pendulumY2

class ResultsPlotter:
    def plotValues(sol:list) -> None:
        t = sol.t
        x = sol.y[0]
        theta = sol.y[2]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

        ax1.set_title('Cart and Pendulum Motion')
        ax1.plot(t, x)
        ax1.set_ylabel('x')

        ax2.plot(t, theta)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('theta')

        plt.show()

    def plotEnergy(sol:list, cartParameters: dict) -> None:
        T = []
        V = []
        E = []

        M = cartParameters['M']
        k = cartParameters['k']

        if cartParameters['dof'] >= 2:
            m = cartParameters['m']
            L = cartParameters['L']
            g = cartParameters['g']
            I = 1 / 12 * m * L**2
            r = L / 2

        for i in range(len(sol.t)):
            x = sol.y[0][i]
            x_dot = sol.y[1][i]

            T.append(0.5 * M * x_dot**2)
            V.append(0.5 * k * x**2)
            E.append(T[i] + V[i])

            if cartParameters['dof'] >= 2:
                theta = sol.y[2][i]
                theta_dot = sol.y[3][i]

                T[i] += 0.5 * m * (x_dot**2 + r**2 * theta_dot**2 + 2 * r * x_dot * theta_dot * np.cos(theta)) + 0.5 * I * theta_dot**2
                V[i] += (-m * g * r * np.cos(theta))
                E[i] = (T[i] + V[i])

            if cartParameters['dof'] >= 3:
                theta2 = sol.y[4][i]
                theta2_dot = sol.y[5][i]

                T[i] += 0.5 * m * ((x_dot + L * theta_dot * np.cos(theta) + r * theta2_dot * np.cos(theta2))**2 + (L * theta_dot * np.sin(theta) + r * theta2_dot * np.sin(theta2))**2 + 0.5 * I * theta2_dot**2)
                V[i] += (-m * g * L * np.cos(theta) - m * g * r * np.cos(theta2))
                E[i] = (T[i] + V[i])

        plt.plot(sol.t, T, label='Kinetic Energy')
        plt.plot(sol.t, V, label='Potential Energy')
        plt.plot(sol.t, E, label='Total Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy vs Time')
        plt.legend()
        plt.show()