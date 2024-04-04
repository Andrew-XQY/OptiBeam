from helper import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with ChangeDirToFileLocation():
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    sys.path.insert(0, full_path)
    import optibeam.simulation as simulation

def animate(i):
    # Calculate means - vary them over the frames
    mu_x = 0.5 + 0.3 * np.sin(2 * np.pi * i / n_frames)
    mu_y = 0.5 + 0.3 * np.cos(2 * np.pi * i / n_frames)
    
    # Calculate standard deviations - vary them over the frames
    sigma_x = 0.05 + 0.02 * np.cos(2 * np.pi * i / n_frames)
    sigma_y = 0.05 + 0.02 * np.sin(2 * np.pi * i / n_frames)
    
    # 2D Gaussian formula
    Z = np.exp(-((X - mu_x)**2 / (2 * sigma_x**2) + (Y - mu_y)**2 / (2 * sigma_y**2)))
    
    # Clearing the previous frame
    ax.clear()
    
    # Display the current frame
    ax.imshow(Z, interpolation='bilinear', cmap='viridis')
    ax.set_title(f"Frame {i+1}")


# Setting up the canvas
fig, ax = plt.subplots()
# Number of frames in the animation
n_frames = 100
# Dimension of the 2D Gaussian
dim = 256


# Generate a meshgrid for the dimensions
x = np.linspace(0, 1, dim)
y = np.linspace(0, 1, dim)
X, Y = np.meshgrid(x, y)

# Creating the animation
ani = FuncAnimation(fig, animate, frames=n_frames, repeat=True)
plt.show()












