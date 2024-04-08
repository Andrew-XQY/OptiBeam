from helper import *
import sys

with ChangeDirToFileLocation():
    full_path = os.path.abspath(os.path.join(os.getcwd(), "../../"))
    sys.path.insert(0, full_path)
    import optibeam.simulation as simulation
    import optibeam.visualization as visualization

dim = (256, 256)
s = simulation.DynamicPatterns(*dim)
visualization.plot_narray(s.generate_gaussian(.155, .666, .15, .1))

