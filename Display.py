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
    def __init__(self, t: list, x: list, theta: list, cartParameters: dict) -> None:
        super().__init__(t, x, cartParameters)
        
        self.theta = theta
        self.pendulumLength = cartParameters["L"]
        
        self.ax.set_ylim(-self.pendulumLength - self.screenHeightBuffer, self.cartHeight + self.screenHeightBuffer)

        self.ax.set_aspect('equal', adjustable='box')

        self.pendulum, = self.ax.plot([], [], color='black', lw=2)

    def _draw_frame(self, i: int) -> None:
        super()._draw_frame(i)
        self._draw_pendulum(i)

    def _draw_pendulum(self, i: int) -> None:
        pendulumX1 = self.x[i]
        pendulumY1 = 0

        pendulumX2 = pendulumX1 + self.pendulumLength * np.sin(self.theta[i])
        pendulumY2 = pendulumY1 + self.pendulumLength * np.cos(self.theta[i])

        self.pendulum.set_data([pendulumX1, pendulumX2], [-pendulumY1, -pendulumY2])

def plotResults(t, x: list) -> None:
    plt.plot(t, x, label='x')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.show()
