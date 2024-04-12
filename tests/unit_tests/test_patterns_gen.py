from conftest import *

canvas = simulation.DynamicPatterns()
canvas._distributions = [simulation.GaussianDistribution(canvas, rotation_radians=0.003) for _ in range(20)] # rotation_radians=0.003

for _ in range(1000):
    canvas.update()
    canvas.plot_canvas()

# import matplotlib.pyplot as plt
# for _ in range(1000):
#     canvas.update()
#     plt.clf()
#     plt.imshow(canvas.macro_pixel(size=8), cmap='viridis')
#     plt.draw()  
#     plt.pause(0.01)


'''TODO'''
# rotation remodelling (decouple centroids movement as a individual function)
# intensity saturation problem (loosing some spatial information? maybe need to fix on the canvas level? depends on the nature and number of distributions?)
# std going too big or too zero after a while
# mimic simplified quadrapole transform in the canvas (develop some possible transformations on canvas level)
# other distributions implementation (Maxwell-Boltzmann, etc)



