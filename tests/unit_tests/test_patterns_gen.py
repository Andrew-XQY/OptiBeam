from conftest import *
import cv2
import time

'''TODO'''
# rotation remodelling (decouple centroids movement as a individual function, apply affine transformation as matrix multiplication)
# intensity saturation problem (loosing some spatial information? maybe need to fix on the canvas level? depends on the nature and number of distributions?)
# mimic simplified quadrapole transform in the canvas? (develop some possible transformations on the canvas level)
# other distributions implementation (Maxwell-Boltzmann, etc)


d = 256
dim = (d, d)
canvas = simulation.DynamicPatterns(*dim)
# canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)] 
canvas._distributions = [simulation.StaticGaussianDistribution(canvas) for _ in range(100)] 

image_arrays = []
for _ in range(10000):
    canvas.update(min_std=0.05, max_std=0.1, max_intensity=130, fade_rate=0.95)  # around 0.95 is good
    canvas.thresholding(1)
    canvas.plot_canvas(cmap='grey')
    #canvas.canvas_pixel_values()
    #break
    time.sleep(1)
    











# d = 256
# dim = (d, d)
# canvas = simulation.DynamicPatterns(*dim)
# canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)] 

# image_arrays = []
# for _ in range(300):
#     canvas.update()
#     canvas.plot_canvas(cmap='grey')
    
#     if _ % 30 == 0:
#         img = canvas.get_image()
#         img = simulation.pixel_value_remap(img)
#         img = simulation.macro_pixel(img, size=int(1024/d))
#         cv2.imwrite("../../ResultsCenter/sync/" + f"{_}.png", img)
        
#     # image_arrays.append(img)

# # visualization.save_as_matplotlib_style_gif(image_arrays, frame_rate=60, save_path='../../ResultsCenter/animation.gif')


# # _______________________ temp _______________________
# width, height = 1024, 1024  # This can be adjusted to your desired size
# gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
# cv2.imwrite('../../ResultsCenter/sync/gradient_image.png', gradient)
# # _______________________ temp _______________________