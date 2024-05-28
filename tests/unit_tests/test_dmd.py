from conftest import *
from ALP4 import *
import numpy as np
import time


# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))

for i in range(10000000):
    # Generate the mosaic image
    img = simulation.create_mosaic_image(size=1024)
    M = simulation.compile_transformation_matrix(image=img) # radians= i*np.pi/180
    img = simulation.apply_transformation_matrix(img, M)
    img = simulation.pixel_value_remap(img, 255)
    # img = np.ones((1024, 1920), dtype=np.uint8)*255
    img = DMD.display_image(img)
    time.sleep(0.3)
    
DMD.end()


