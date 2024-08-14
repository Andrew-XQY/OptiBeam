from conftest import *
import cv2
import time

'''TODO'''
# rotation remodelling (decouple centroids movement as a individual function, apply affine transformation as matrix multiplication)
# intensity saturation problem (loosing some spatial information? maybe need to fix on the canvas level? depends on the nature and number of distributions?)
# mimic simplified quadrapole transform in the canvas? (develop some possible transformations on the canvas level)

d = 256
dim = (d, d)
canvas = simulation.DynamicPatterns(*dim)
# canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)] 
canvas._distributions = [simulation.StaticGaussianDistribution(canvas) for _ in range(100)] 

image_arrays = []
for _ in range(10000):
    canvas.update(std_1=0.03, std_2=0.2, max_intensity=100,
                  fade_rate=0.96, distribution='other')  # around 0.95 looks good std_1=0.15, std_2=0.12
    canvas.thresholding(1)
    # canvas.canvas_pixel_values()
    # break
    canvas.plot_canvas(cmap='grey')
    time.sleep(1)
    













# ---------------------- Dynamic Multiple Guassian distribution ----------------------
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