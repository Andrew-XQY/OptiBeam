from conftest import *
from ALP4 import *
import time

# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
canvas = simulation.DynamicPatterns(*(512, 512))
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(10)]

for i in range(1000000):
    canvas.update()
    img = canvas.get_image()
    img = simulation.pixel_value_remap(img)
    img = simulation.macro_pixel(img, size=2)
    DMD.display_image(img)
    
    #time.sleep(0.2)
    
DMD.end()































