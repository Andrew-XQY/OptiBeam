from helper import *
import sys
import random

with ChangeDirToFileLocation():
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    sys.path.insert(0, full_path)
    import optibeam.simulation as simulation
    import optibeam.visualization as visualization

dim = (256, 256)
canvas = simulation.DynamicPatterns(*dim)

# guassian = simulation.GaussianDistribution(canvas)
# guassian.add_transformation(simulation.quadrupole_transform)
# canvas.distributions = [guassian]

canvas._distributions = [simulation.GaussianDistribution(canvas, rotation_radians=0.003) for _ in range(15)]

for _ in range(1000):
    canvas.update()
    canvas.plot_canvas()
    








