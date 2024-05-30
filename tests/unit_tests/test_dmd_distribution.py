from conftest import *
from ALP4 import *
import time

# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
dim = 1024
canvas = simulation.DynamicPatterns(*(dim, dim))
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(5)]

for i in range(1000000):
    canvas.update()
    img = canvas.get_image()
    img = simulation.pixel_value_remap(img)
    img = simulation.macro_pixel(img, size=int(1024/dim))
    DMD.display_image(img)
    
DMD.end()































