from conftest import *

'''TODO'''
# rotation remodelling (decouple centroids movement as a individual function, apply affine transformation as matrix multiplication)
# intensity saturation problem (loosing some spatial information? maybe need to fix on the canvas level? depends on the nature and number of distributions?)
# mimic simplified quadrapole transform in the canvas? (develop some possible transformations on the canvas level)
# other distributions implementation (Maxwell-Boltzmann, etc)

dim = (256, 256)
canvas = simulation.DynamicPatterns(*dim)
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(20)] 

image_arrays = []
for _ in range(300):
    canvas.update()
    canvas.plot_canvas()
    img = canvas.get_image()
    image_arrays.append(img)


visualization.save_as_matplotlib_style_gif(image_arrays, frame_rate=60, save_path='../../ResultsCenter/animation.gif')


