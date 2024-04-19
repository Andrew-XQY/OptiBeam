from conftest import *

# for i in range(360):
#     # Generate the mosaic image
#     img = simulation.create_mosaic_image(size=512)
#     M = simulation.compile_transformation_matrix(image=img, radians= i*np.pi/180)
#     img = simulation.apply_transformation_matrix(img, M)

#     img = dmd.pad_image(img, 1024, 1024)
#     plt.imshow(img, cmap='gray')
#     plt.draw()
#     plt.pause(0.01)
#     plt.clf()


canvas = simulation.DynamicPatterns(*(64, 64))
canvas._distributions = [simulation.GaussianDistribution(canvas, rotation_radians=0.003) for _ in range(10)] # rotation_radians=0.003
    
for i in range(100):
    # Generate the mosaic image
    canvas.update()
    img = simulation.pixel_value_remap(canvas.get_image(), 255)
    img = simulation.macro_pixel(img, size=16)
    img = dmd.pad_image(img, 1024, 1920)
    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.01)
    plt.clf()


print(np.max(img), np.min(img))    
