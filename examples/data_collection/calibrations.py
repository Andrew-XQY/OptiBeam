from conf import *
from ALP4 import *
import datetime, time
import cv2
from multiprocessing import Process, Event, Queue

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

# DMD process
def dmd_process(stop_event, queue, conf=None):
    DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
    calibrator = simulation.CornerBlocksCalibrator(block_size=32)
    while not stop_event.is_set():
        if not queue.empty():
            calibrator.set_special(queue.get())
        calibrator.generate_blocks()
        img = calibrator.canvas
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim']/img.shape[0])) 
        DMD.display_image(dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation']))
        time.sleep(1)
    DMD.end()
    
def dmd_process_1(stop_event, queue, conf=None):
    DMD = dmd.ViALUXDMD(ALP4(version = '4.3'))
    while not stop_event.is_set():
        img = np.ones((256, 256)) * 100
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim']/img.shape[0])) 
        DMD.display_image(dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation']))
        time.sleep(1)
    DMD.end()

# Camera process
def camera_process(stop_event, queue, conf=None):
    def on_trackbar(val):
        queue.put(val)  # Send the trackbar value to the queue

    cv2.namedWindow("Camera Feed")
    cv2.createTrackbar("Special", "Camera Feed", 0, 4, on_trackbar)
    gen = camera_generator(stop_event)
    for frame in gen:
        if stop_event.is_set():
            break
        # Display the camera frame
        frame = processing.crop_image_from_coordinates(frame, conf['crop_areas'])
        l, r = utils.split_image(frame)
        ratio = processing.get_coupling_ratio(input_img=l, output_img=r)
        ratio = round(ratio, 2)
        frame = utils.scale_image(frame, 2)
        cv2.putText(frame, 'Coupling Ratio: ' + str(ratio), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'esc' to stop
            stop_event.set()
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # ============================
    # Dataset Parameters
    # ============================
    conf = {
        'dmd_dim': 1024,  # DMD working square area resolution
        'dmd_rotation': 38,  # DMD rotation angle for image orientation correction
        'dmd_bitDepth': 8,  # DMD bit depth
        'dmd_picture_time': 20000,  # DMD picture time in microseconds, corresponds to 50 Hz
        'crop_areas': [((869, 612), (1003, 746)), ((2295, 2), (3487, 1194))]    # crop areas for the camera images
    }

    # Create a stop event for graceful termination
    stop_event = Event()
    queue = Queue()

    # Create and start processes
    camera_proc = Process(target=camera_process, args=(stop_event, queue, conf))
    dmd_proc = Process(target=dmd_process, args=(stop_event, queue, conf))

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

    