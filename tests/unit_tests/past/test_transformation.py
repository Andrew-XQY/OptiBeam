from conftest import *


image = simulation.create_mosaic_image(n=4)
for i in range(10000):
    M = simulation.compile_transformation_matrix(image=image, radians= i*0.01*np.pi/180)
    image = simulation.apply_transformation_matrix(image, M)
    plt.imshow(image, cmap='gray')
    plt.draw()
    plt.pause(0.01)
    plt.clf()
    
    
