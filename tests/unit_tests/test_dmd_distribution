from conftest import *
from ALP4 import *

# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
canvas = simulation.DynamicPatterns(*(64, 64))
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)]

for i in range(1000):
    canvas.update()
    img = canvas.get_image()
    img = simulation.pixel_value_remap(img)
    img = simulation.macro_pixel(img, size=16)
    DMD.display_image(img)
    
DMD.end()































