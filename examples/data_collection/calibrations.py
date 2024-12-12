from conf import *
from ALP4 import *
import datetime, time
import cv2
from multiprocessing import Process, Event

# ============================
# Dataset Parameters
# ============================
conf = {
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': 47+90,  # DMD rotation angle for image orientation correction
    'dmd_bitDepth': 8,  # DMD bit depth
    'dmd_picture_time': 100000,  # DMD picture time in microseconds, corresponds to 50 Hz
    'crop_areas': [((871, 434), (1027, 590)), ((2848, 440), (3042, 634))]  # crop areas for the camera images
}

# ============================
# Multiprocessing DMD image display and camera capture
# ============================
def camera_generator(stop_event):
    MANAGER = camera.MultiBaslerCameraManager()
    MANAGER._initialize_cams()
    MANAGER.flip = True
    free_run_gen = MANAGER.free_run()  # Create the generator from MANAGER's free_run method
    try:
        while not stop_event.is_set():
            try:
                yield next(free_run_gen)
            except StopIteration:
                break
    finally:
        stop_event.set()
    MANAGER.end()

def dmd_generator(stop_event):
    corner_gen = simulation.corner_blocks_generator()
    while not stop_event.is_set():
        yield next(corner_gen)

# Camera process
def camera_process(stop_event, conf=None):
    gen = camera_generator(stop_event)
    for frame in gen:
        if stop_event.is_set():
            break
        # Display the camera frame
        frame = processing.crop_image_from_coordinates(frame, conf['crop_areas'])
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' to stop
            stop_event.set()
            break
    cv2.destroyAllWindows()

# DMD process
def dmd_process(stop_event, conf=None):
    DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
    gen = dmd_generator(stop_event)
    for img in gen:
        if stop_event.is_set():
            break
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim']/img.shape[0])) 
        DMD.display_image(dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation']))
        time.sleep(1)
    DMD.end()


if __name__ == "__main__":
    # Create a stop event for graceful termination
    stop_event = Event()

    # Create and start processes
    camera_proc = Process(target=camera_process, args=(stop_event, conf))
    dmd_proc = Process(target=dmd_process, args=(stop_event, conf))

    camera_proc.start()
    dmd_proc.start()

    try:
        # Main thread waits for processes to finish or interrupt
        while not stop_event.is_set():
            pass
    except KeyboardInterrupt:
        print("Stopping program...")
        stop_event.set()  # Signal all processes to stop

    # Ensure all processes terminate cleanly
    camera_proc.join()
    dmd_proc.join()

    