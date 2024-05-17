from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


class CartAnimation(animation.TimedAnimation):
    def __init__(self, t, x, cartParameters):
        self.t = t
        self.x = x

        self.cartWidth = 2
        self.cartHeight = 1
        self.pendulumLength = cartParameters["L"]

        self.screenWidthBuffer = 2
        self.screenHeightBuffer = 2

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-(max(abs(self.x)) + self.screenWidthBuffer + self.cartWidth // 2), max(abs(self.x)) + self.screenWidthBuffer + self.cartWidth // 2)
        self.ax.set_ylim(-self.screenHeightBuffer, self.cartHeight + self.screenHeightBuffer)

        self.ax.set_aspect('equal', adjustable='box')

        self.cart = plt.Rectangle((self.x[0] - self.cartWidth // 2, 0), self.cartWidth, self.cartHeight, fill=True, color='blue')

        self.ax.add_patch(self.cart)

        animation.TimedAnimation.__init__(self, self.fig, interval=(t[1] - t[0]) * 1000, blit=True)

    def _draw_frame(self, i):
        self._draw_cart(i)
    
    def _draw_cart(self, i):
        self.cart.set_xy([self.x[i] - self.cartWidth // 2, 0])

    def new_frame_seq(self):
        return iter(range(len(self.t)))

    def show(self):
        plt.show()

    def save(self, filename):
        self.save(filename, writer='imagemagick')

class CartAndPendulumAnimation(CartAnimation):
    def __init__(self, t, x, theta, cartParameters):
        super().__init__(t, x, cartParameters)
        
        self.theta = theta

        self.ax.set_ylim(-self.pendulumLength - self.screenHeightBuffer, self.cartHeight + self.screenHeightBuffer)

        self.ax.set_aspect('equal', adjustable='box')

        self.pendulum, = self.ax.plot([], [], color='black', lw=2)

    def _draw_frame(self, i):
        super()._draw_frame(i)
        self._draw_pendulum(i)

    def _draw_pendulum(self, i):
        pendulumX1 = self.x[i]
        pendulumY1 = 0

        pendulumX2 = pendulumX1 + self.pendulumLength * np.sin(self.theta[i])
        pendulumY2 = pendulumY1 + self.pendulumLength * np.cos(self.theta[i])

        self.pendulum.set_data([pendulumX1, pendulumX2], [-pendulumY1, -pendulumY2])




def plotResults(t, x):
    plt.plot(t, x, label='x')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.show()
