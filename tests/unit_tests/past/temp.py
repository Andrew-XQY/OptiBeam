from conftest import *
import cv2
import datetime

manager = camera.MultiBaslerCameraManager()
manager.synchronization()

save_path = "../../ResultsCenter/sync/"
# image =  manager.perodically_scheduled_action_command()
for _ in range(20):
    image = manager.schedule_action_command(int(2000 * 1e6))
    if image is not None:
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cv2.imwrite(save_path + filename + '.png', image)

manager.end()
