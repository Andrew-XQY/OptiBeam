from conftest import *
from ALP4 import *
import datetime
import cv2
import threading


MAX_DMD_DIM = 1024
class DataCollection:
    def __init__(self):
        self.DMD = dmd.ViALUXDMD(ALP4(version = '4.3')) # Load the Vialux .dll
        self.dim = 1024
        self.canvas = simulation.DynamicPatterns(*(self.dim, self.dim))
        self.canvas._distributions = [simulation.GaussianDistribution(self.canvas) for _ in range(10)]
        self.stop = False
        self.img = None
        
    def dmd_refreshing(self):
        while not self.stop:
            self.canvas.update()
            img = self.canvas.get_image()
            img = simulation.pixel_value_remap(img)
            img = simulation.macro_pixel(img, size=int(1024/self.dim))
            self.img = img
            # img = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
            scale = 1 / np.sqrt(2)
            center = (MAX_DMD_DIM // 2, MAX_DMD_DIM // 2)
            M = cv2.getRotationMatrix2D(center, 45, scale)
            img = cv2.warpAffine(img, M, (MAX_DMD_DIM, MAX_DMD_DIM), 
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=(0, 0, 0))
            self.DMD.display_image(img)
            # time.sleep(0.1)
            
    def end(self):
        self.DMD.end()

@utils.add_progress_bar
def generate_data(n : int=10):
    return range(n)
        
# Create and start the thread
data = DataCollection()
thread = threading.Thread(target=data.dmd_refreshing)
thread.start()

manager = camera.MultiBaslerCameraManager()
manager.synchronization()
save_path = "../../ResultsCenter/sync/"
for _ in generate_data():
    image = manager.schedule_action_command(int(3000 * 1e6))
    if image is not None:
        resized_image = cv2.resize(data.img, (int(image.shape[1]//2),image.shape[0]))
        rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
        three_imgs = np.hstack((rotated_image, image))
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cv2.imwrite(save_path + filename + '.png', three_imgs)

data.stop = True
thread.join()
data.end()
manager.end()







