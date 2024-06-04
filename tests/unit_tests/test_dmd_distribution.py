from conftest import *
from ALP4 import *
import time
import cv2

# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
dim = 512
canvas = simulation.DynamicPatterns(*(dim, dim))
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(10)]

for i in range(1000000):
    canvas.update()
    img = canvas.get_image()
    img = simulation.pixel_value_remap(img)
    img = simulation.macro_pixel(img, size=int(1024/dim))
    # img = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
    scale = 1 / np.sqrt(2)
    center = (1024 // 2, 1024 // 2)
    M = cv2.getRotationMatrix2D(center, 45, scale)
    img = cv2.warpAffine(img, M, (1024, 1024), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))

    DMD.display_image(img)
    # time.sleep(0.1)
    
DMD.end()































