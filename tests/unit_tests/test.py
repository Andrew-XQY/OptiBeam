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
canvas.distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)]

for _ in range(1000):
    canvas.update()
    canvas.plot_canvas()
    
# pattern = guassian.generate_2d_gaussian()
# visualization.plot_narray(pattern)








