from conftest import *

'''TODO'''
# rotation remodelling (decouple centroids movement as a individual function, apply affine transformation as matrix multiplication)
# intensity saturation problem (loosing some spatial information? maybe need to fix on the canvas level? depends on the nature and number of distributions?)
# mimic simplified quadrapole transform in the canvas? (develop some possible transformations on the canvas level)
# other distributions implementation (Maxwell-Boltzmann, etc)

dim = (256, 128)
canvas = simulation.DynamicPatterns(*dim)
canvas._distributions = [simulation.GaussianDistribution(canvas) for _ in range(1)] 

for _ in range(5000):
    canvas.update()
    canvas.plot_canvas()




