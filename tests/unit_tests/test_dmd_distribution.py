from conftest import *
from ALP4 import *
import time
import cv2

conf = {
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': -45,  # DMD rotation angle for image orientation correction
}

# Load the Vialux .dll
DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
dim = 512
canvas = simulation.DynamicPatterns(*(dim, dim))
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(10)]



for i in range(1000000):
    # canvas.update()
    # img = canvas.get_image()
    # img = simulation.pixel_value_remap(img)
    # img = simulation.macro_pixel(img, size=int(1024/dim))
    
    # img = np.ones((256, 256)) * 100
    # img = simulation.macro_pixel(img, size=int(1024/dim))
    # # img = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
    # scale = 1 / np.sqrt(2)
    # center = (1024 // 2, 1024 // 2)
    # M = cv2.getRotationMatrix2D(center, 42, scale)
    # img = cv2.warpAffine(img, M, (1024, 1024), 
    #                                borderMode=cv2.BORDER_CONSTANT, 
    #                                borderValue=(0, 0, 0))

    # DMD.display_image(img)
    
    def create_solid_circle(size=256, intensity=255):
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 2
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        img = np.zeros((size, size))
        img[mask] = intensity
        return img
    
    img = np.ones((256, 256)) * 255
    # img = create_solid_circle(256, 255)
    # img = simulation.generate_upward_arrow()
    img = simulation.macro_pixel(img, size=int(conf['dmd_dim']/img.shape[0])) 
    DMD.display_image(dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation'])) 

    time.sleep(0.5)
    
DMD.end()































