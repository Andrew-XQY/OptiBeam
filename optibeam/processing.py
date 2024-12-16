import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .utils import crop_images, resize_image, join_images, resize_image_high_quality



# ----------------------------- dataset preparation -----------------------------

# class HDF5DatasetWriter:
#     def __init__(self, dims, output_path, data_key="images", buf_size=1000):
#         # Check if the output path exists
#         if os.path.exists(output_path):
#             raise ValueError("The supplied 'output_path' already exists and cannot be overwritten. "
#                              "Please specify a different 'output_path'.")

#         # Open the HDF5 database for writing and create two datasets: one to store the images/features and another
#         # to store the class labels
#         self.db = h5py.File(output_path, "w")
#         self.data = self.db.create_dataset(data_key, dims, dtype="float")
#         self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

#         # Store the buffer size and initialize the buffer itself
#         self.buf_size = buf_size
#         self.buffer = {"data": [], "labels": []}
#         self.idx = 0

#     def add(self, rows, labels):
#         # Add the rows and labels to the buffer
#         self.buffer["data"].extend(rows)
#         self.buffer["labels"].extend(labels)

#         # Check to see if the buffer needs to be flushed to disk
#         if len(self.buffer["data"]) >= self.buf_size:
#             self.flush()

#     def flush(self):
#         # Write the buffers to disk then reset the buffer
#         i = self.idx + len(self.buffer["data"])
#         self.data[self.idx:i] = self.buffer["data"]
#         self.labels[self.idx:i] = self.buffer["labels"]
#         self.idx = i
#         self.buffer = {"data": [], "labels": []}

#     def store_class_labels(self, class_labels):
#         # Create a dataset to store the actual class label names


# ----------------------------- Functionality visualization -----------------------------

def add_grid(image: np.ndarray, partitions: int, grid_color: int = 255) -> np.ndarray:
    """
    Adds a dashed square grid to an image, with grid color, spacing, and thickness determined dynamically.

    Args:
        image (np.ndarray): The input image in narray format (2D or 3D).
        partitions (int): The number of partitions for the grid.
        grid_color (int): The color of the grid lines (default is 255 for white).

    Returns:
        np.ndarray: The modified image with a dashed square grid.
    """
    if len(image.shape) == 3:  # For colored images
        height, width, channels = image.shape
    elif len(image.shape) == 2:  # For grayscale images
        height, width = image.shape
        channels = 1  # Treat grayscale images as single-channel for grid overlay
    else:
        raise ValueError("Image should be either a 2D grayscale or 3D color image.")
    
    # Dash line properties
    dash_length = 2  # Length of each dash
    gap_length = 2   # Length of the gap between dashes
    dash_pattern = dash_length + gap_length

    # Grid thickness
    grid_thickness = 1  # Change this value to control the grid thickness

    # Determine the step size based on the shorter dimension
    shorter_dim = min(height, width)
    step = shorter_dim // partitions  # Equal grid step size for both dimensions

    # Create a copy of the image to modify
    grid_image = image.copy()
    
    # Add dashed horizontal lines
    for y in range(0, height, step):
        for x in range(0, width, dash_pattern):  # Apply the dash pattern
            if channels > 1:  # For colored images
                grid_image[max(0, y-grid_thickness//2):y+grid_thickness//2+1, x:x+dash_length, :] = grid_color
            else:  # For grayscale images
                grid_image[max(0, y-grid_thickness//2):y+grid_thickness//2+1, x:x+dash_length] = grid_color

    # Add dashed vertical lines
    for x in range(0, width, step):
        for y in range(0, height, dash_pattern):  # Apply the dash pattern
            if channels > 1:  # For colored images
                grid_image[y:y+dash_length, max(0, x-grid_thickness//2):x+grid_thickness//2+1, :] = grid_color
            else:  # For grayscale images
                grid_image[y:y+dash_length, max(0, x-grid_thickness//2):x+grid_thickness//2+1] = grid_color

    return grid_image


def calculate_pixel_sum(image: np.ndarray, square_values: bool = False) -> float:
    """
    Calculate the accumulative total pixel value of an image.

    Args:
        image (np.ndarray): Input image as a single-channel NumPy array.
        square_values (bool): If True, square the pixel values before summing.

    Returns:
        float: The accumulative total pixel value.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image must be a NumPy array.")

    if square_values:
        return np.sum(np.square(image))
    else:
        return np.sum(image)

class IntensityMonitor:
    def __init__(self, buffer_size=100):
        """
        Initialize the IntensityMonitor object.

        Args:
            buffer_size (int): Maximum size of the intensity buffer.
        """
        self.buffer = deque(maxlen=buffer_size)

    def add_image(self, image: np.ndarray, square_values: bool = False) -> None:
        """
        Process an image to calculate its intensity and add it to the buffer.

        Args:
            image (np.ndarray): Input image as a single-channel NumPy array.
            square_values (bool): If True, square the pixel values before summing.
        """
        try:
            intensity = calculate_pixel_sum(image, square_values)
        except Exception:
            intensity = 0  # Default to 0 if the calculation fails
        self.buffer.append(intensity)

    def plot_intensity(self, channel=1) -> np.ndarray:
        """
        Generate a time-series plot of the intensity buffer.

        Args:
            channel (int): Number of channels (1 for grayscale, 3 for RGB-like dummy expansion).

        Returns:
            np.ndarray: The plotted Matplotlib figure as a NumPy array.
        """
        fig, ax = plt.subplots()
        ax.plot(range(len(self.buffer)), list(self.buffer), label="Intensity Sum")
        ax.set_title("Intensity Over Time")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Intensity Sum")
        ax.legend()
        ax.grid()

        # Convert the plot to a NumPy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))

        if channel == 1:
            # Convert to grayscale by averaging channels
            image = np.mean(image, axis=2).astype('uint8')

        plt.close(fig)  # Close the figure to free memory
        return image

    def demo(self, channel=1):
        """
        Display the intensity plot.

        Args:
            channel (int): Number of channels (1 for grayscale, 3 for RGB-like dummy expansion).
        """
        image = self.plot_intensity(channel=channel)
        plt.imshow(image, cmap="gray" if channel == 1 else None)
        plt.axis('off')
        plt.show()


# ----------------------------- image processing -----------------------------

def apply_threshold(image: np.ndarray, threshold=5) -> np.ndarray:
    """
    Apply a threshold to an image array. Pixels below the threshold are set to 0.
    If the image is normalized (0 to 1), the threshold is also normalized.

    Parameters:
    - image: numpy array, the input image.
    - threshold: float, the threshold value.

    Returns:
    - numpy array, the thresholded image.
    """
    # Check if the image is normalized
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.min() >= 0 and image.max() <= 1:
            normalized_threshold = threshold / 255.0
        else:
            normalized_threshold = threshold
    else:
        normalized_threshold = threshold
    thresholded_image = np.where(image >= normalized_threshold, image, 0)
    return thresholded_image


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure that the input image is a single-channel grayscale image.
    """
    # Check if the image has 2 dimensions (grayscale) or more
    if len(image.shape) > 2:
        # Convert from BGR or BGRA to Grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # Image is already single-channel grayscale
        return image
    else:
        raise ValueError("Unsupported image format")



# ----------------------------- image sample post processing -----------------------------

def crop_images_from_clicks(click_list, image):
    """
    Crop images based on a list of click positions that define the rectangle corners.
    
    Args:
    click_list (list of tuples): List containing pairs of tuples. Each pair defines
                                 the top left and bottom right corners of a rectangle.
    image (numpy.ndarray): The image from which to crop the rectangles.

    Returns:
    list of numpy.ndarray: List containing the cropped images.
    """
    cropped_images = []
    
    # Iterate through the list of tuples; each pair forms one rectangle
    for i in range(0, len(click_list)):
        top_left = click_list[i][0]
        bottom_right = click_list[i][1]
        
        # Crop the image using numpy slicing
        cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cropped_images.append(cropped)
    
    return cropped_images


def select_crop_areas_corner(original_image, num, scale_factor=1):
    
    # Check if the image is single-channel and convert to BGR
    if len(original_image.shape) == 2:  # Grayscale image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Helper variables
    points = []
    squares = []
    max_x, max_y = original_image.shape[1], original_image.shape[0]

    def mouse_click(event, x, y, flags, param):
        nonlocal points, squares

        # Adjust click position to original image scale
        orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)

        # Record the click positions
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2 * num:  # Reset if the previous set is complete
                points = []
                squares = []
                image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                cv2.imshow('Image', image)

            points.append((orig_x, orig_y))

            # Check if we can form a square
            if len(points) % 2 == 0:
                point1 = points[-2]
                point2 = points[-1]

                # Calculate the center and the side length of the square
                center_x = (point1[0] + point2[0]) // 2
                center_y = (point1[1] + point2[1]) // 2
                side = max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))

                # Calculate the top-left and bottom-right corners of the square
                top_left = (center_x - side // 2, center_y - side // 2)
                bottom_right = (center_x + side // 2, center_y + side // 2)

                # Constrain points to valid image bounds
                top_left = (max(0, min(top_left[0], max_x)), max(0, min(top_left[1], max_y)))
                bottom_right = (max(0, min(bottom_right[0], max_x)), max(0, min(bottom_right[1], max_y)))

                squares.append((top_left, bottom_right))

            # Redraw the image with squares/points
            image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            for square in squares:
                scaled_square = [(int(pt[0] * scale_factor), int(pt[1] * scale_factor)) for pt in square]
                cv2.rectangle(image, scaled_square[0], scaled_square[1], (0, 255, 0), 1)
            if len(points) % 2 == 1:
                cv2.circle(image, (int(points[-1][0] * scale_factor), int(points[-1][1] * scale_factor)), 1, (0, 0, 255), -1)
            cv2.imshow('Image', image)

    # Scale and display the image
    scaled_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_click)
    cv2.imshow('Image', scaled_image)
    # Handle the window until ESC is pressed
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    return squares



def detect_round_and_draw_bounds(image: np.ndarray) -> np.ndarray:
    """
    Detect round shapes in the image and draw the bounds around them.
    Have to carefully tune the parameters for the Hough Circle Transform.
    
    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with the bounds drawn around the detected circles
    """
    image = ensure_grayscale(image)
    # Threshold the image to isolate the dots
    _, thresh = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    # Apply morphological dilation to merge the dots
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Use Hough Circle Transform to detect the circle
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, minDist=250,
                               param1=220, param2=10, minRadius=400, maxRadius=1000)
    # Draw circles that are detected. 
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # Draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    return cimg


def select_crop_areas_center(original_image: np.ndarray, num: int, scale_factor: float=1, autodetect: bool=False) -> list:
    # Check if the image is single-channel and convert to BGR
    if len(original_image.shape) == 2:  # Grayscale image
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
    # Helper variables
    points = []
    squares = []
    
    if autodetect:
        original_image = detect_round_and_draw_bounds(original_image)

    def mouse_click(event, x, y, flags, param):
        nonlocal points, squares
        max_x, max_y = original_image.shape[1], original_image.shape[0]
        # Adjust click position to original image scale
        orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)

        # Record the click positions
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2 * num:  # Reset if previous set is complete
                points = []
                squares = []
                image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                cv2.imshow('Image', image)

            points.append((orig_x, orig_y))

            # Check if we can form a square
            if len(points) % 2 == 0:
                center = points[-2]
                edge_point = points[-1]
                side_length = max(abs(edge_point[0] - center[0]), abs(edge_point[1] - center[1]))
                top_left = (center[0] - side_length, center[1] - side_length)
                bottom_right = (center[0] + side_length, center[1] + side_length)
                top_left = (max(0, min(top_left[0], max_x)), max(0, min(top_left[1], max_y)))
                bottom_right = (max(0, min(bottom_right[0], max_x)), max(0, min(bottom_right[1], max_y)))
                squares.append((top_left, bottom_right))

            # Redraw the image with squares
            image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            for square in squares:
                cv2.rectangle(image, (int(square[0][0] * scale_factor), int(square[0][1] * scale_factor)),
                              (int(square[1][0] * scale_factor), int(square[1][1] * scale_factor)), (0, 255, 0), 1)
            if len(points) % 2 == 1:
                cv2.circle(image, (int(points[-1][0] * scale_factor), int(points[-1][1] * scale_factor)), 1, (0, 0, 255), -1)
            cv2.imshow('Image', image)

    # Scale and display the image
    
    scaled_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_click)
    cv2.imshow('Image', scaled_image)

    # Handle the window until ESC is pressed
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    return squares


def crop_image_from_coordinates(image: np.ndarray, crop_areas: list, DIM=(256, 256)) -> np.ndarray:
    img1, img2 = crop_images(image, regions=crop_areas)
    img1 = resize_image(img1, new_dimensions=DIM)
    img2 = resize_image(img2, new_dimensions=DIM)
    image = join_images([img1, img2])
    image = (image * 255).astype(np.uint8)
    return image
