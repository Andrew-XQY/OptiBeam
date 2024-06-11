import cv2
import numpy as np

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
    # Helper variables
    points = []
    rectangles = []
    max_x, max_y = original_image.shape[1], original_image.shape[0]

    def mouse_click(event, x, y, flags, param):
        # Access the points list
        nonlocal points, rectangles

        # Adjust click position to original image scale
        orig_x, orig_y = int(x / scale_factor), int(y / scale_factor)

        # Record the click positions
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2 * num:  # Reset if previous set is complete
                points = []
                rectangles = []
                image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                cv2.imshow('Image', image)

            points.append((orig_x, orig_y))

            # Check if we can form a rectangle
            if len(points) % 2 == 0:
                points[-2] = (max(0, min(points[-2][0], max_x)), max(0, min(points[-2][1], max_y)))
                points[-1] = (max(0, min(points[-1][0], max_x)), max(0, min(points[-1][1], max_y)))
                rectangles.append((points[-2], points[-1]))

            # Redraw the image with rectangles/points
            image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            for rect in rectangles:
                scaled_rect = [(int(pt[0] * scale_factor), int(pt[1] * scale_factor)) for pt in rect]
                cv2.rectangle(image, scaled_rect[0], scaled_rect[1], (0, 255, 0), 1)
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
    return rectangles


def select_crop_areas_center(original_image, num, scale_factor=1):
    # Helper variables
    points = []
    squares = []

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




