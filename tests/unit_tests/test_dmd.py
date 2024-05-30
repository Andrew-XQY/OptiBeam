from conftest import *
from ALP4 import *
import numpy as np
import time

# _______________________ test rotating mosaic _______________________
# DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
# for i in range(10000000):
#     # Generate the mosaic image
#     img = simulation.create_mosaic_image(size=1024)
#     M = simulation.compile_transformation_matrix(image=img) # radians= i*np.pi/180
#     img = simulation.apply_transformation_matrix(img, M)
#     img = simulation.pixel_value_remap(img, 255)
#     #img = np.ones((1024, 1024), dtype=np.uint8)*255
#     img = DMD.display_image(img)
#     time.sleep(0.3)
# DMD.end()





# _______________________ test contour line problem _______________________

DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
for i in range(10000000):
    # Generate the mosaic image
    width, height = 1024, 1024  # This can be adjusted to your desired size
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    M = simulation.compile_transformation_matrix(image=gradient, radians= i*np.pi/180) # radians= i*np.pi/180
    img = simulation.apply_transformation_matrix(gradient, M)
    img = DMD.display_image(img)
    time.sleep(0.3)
DMD.end()




