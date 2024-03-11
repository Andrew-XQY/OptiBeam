'''
This module contains the functions to evaluate the performance of the model
Including:
1. Image to parameter (speckle pattern -> beam centroids (horizontal, vertical), beam widths).
2. Common Image Quality Metrics (PSNR, SSIM, MSE, etc.) for image reconstruction.
'''

from .utils import *
from scipy.optimize import curve_fit
from scipy.stats import norm


# ------------------- data processing -------------------

def subtract_min_from_array(arr):
    """
    Subtract the minimum value from all elements in a 2D numpy array.
    Parameters:
    arr (np.ndarray): A 2D numpy array.
    Returns:
    np.ndarray: A 2D numpy array with the minimum value subtracted from all elements.
    """
    min_value = np.min(arr)
    return arr - min_value


# ------------------- Image to Parameters Metrics -------------------

# Define the Gaussian function
def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))



def fit_gaussian(x, y):
    """
    Gaussian fit (Least Squares Fitting) 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    """
    # Initial guesses for fitting parameters: A, mu, sigma
    initial_guesses = [max(y), np.mean(x), np.std(x)]
    params, covariance = curve_fit(gaussian, x, y, p0=initial_guesses)
    y_fit = gaussian(x, *params)
    return y_fit, params



def beam_params(img, func=subtract_minimum):
    """
    Input image -> beam parameters (beam centroids, beam widths), not normalized. Two 1D Gaussian fits are used.
    img: 2d numpy array representing the image
    func: function, optional, used to process the histogram data, e.g. minmax_normalization
    """
    horizontal_x = np.arange(len(img[0]))
    vertical_x = np.arange(len(img))
    horizontal_histogram = np.sum(img, axis=0)
    vertical_histogram = np.sum(img, axis=1)
    if func:
        horizontal_histogram = func(horizontal_histogram)
        vertical_histogram = func(vertical_histogram)
    try:
        _, v_params = fit_gaussian(horizontal_x, horizontal_histogram) # assume it is a Gaussian beam                        
        _, h_params = fit_gaussian(vertical_x, vertical_histogram)
        res = {"horizontal_centroid" : v_params[1], "vertical_centroid" : h_params[1],
            "horizontal_width" : v_params[2], "vertical_width" : h_params[2]}
    except: # if the fitting fails, return the center of the image, zero sigma can be seen as a indicator
        res = {"horizontal_centroid" : len(img[0]) // 2, "vertical_centroid" : len(img) // 2,
            "horizontal_width" : 0, "vertical_width" : 0}
    return res



# 2D Gaussian fit (Least Squares Fitting)
def gaussian_2d(xy, A, x0, y0, sigma_x, sigma_y):
    x, y = xy
    inner = ((x - x0) ** 2 / (2 * sigma_x ** 2)) + ((y - y0) ** 2 / (2 * sigma_y ** 2))
    return A * np.exp(-inner)


def fit_2d_gaussian(image):
    # Generate x and y indices
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Flatten the x, y indices and image for fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    xy_flat = np.vstack((x_flat, y_flat))
    image_flat = image.flatten()
    
    # Initial guesses for fitting parameters
    A_guess = np.max(image)
    x0_guess = np.mean(x_flat)
    y0_guess = np.mean(y_flat)
    sigma_x_guess = np.std(x_flat)
    sigma_y_guess = np.std(y_flat)
    initial_guess = (A_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess)
    
    # Fit the 2D Gaussian
    try:
        popt, pcov = curve_fit(gaussian_2d, xy_flat, image_flat, p0=initial_guess)
        res = {"horizontal_centroid" : popt[1], "vertical_centroid" : popt[2],
                "horizontal_width" : popt[3], "vertical_width" : popt[4]}
    except: # if the fitting fails, return the center of the image, zero sigma can be seen as a indicator
        res = {"vertical_centroid" : len(image[0]) // 2, "horizontal_centroid" : len(image) // 2,
            "vertical_width" : 0, "horizontal_width" : 0}
    return res  # Returns the optimized parameters (A, x0, y0, sigma_x, sigma_y)






# ------------------- image illustraion/visualization functions -------------------


def plot_gaussian_fit(image):
    """
    Plot the horizontal and vertical histograms of the image, and the Gaussian fit.
    input: 2d numpy array representing the image
    output: None
    """
    # Calculate vertical/horizontal histogram
    horizontal_histogram = subtract_minimum(np.sum(image, axis=1)[::-1])
    vertical_histogram = subtract_minimum(np.sum(image, axis=0))

    horizontal_x = np.arange(len(horizontal_histogram))
    vertical_x = np.arange(len(vertical_histogram))

    params = beam_params(image)
    h_mu = params["vertical_centroid"]  # history mistake
    v_mu = params["horizontal_centroid"]
    h_sigma = params["vertical_width"]
    v_sigma = params["horizontal_width"]
    
    try:
        horizontal_fit, _ = fit_gaussian(horizontal_x, horizontal_histogram)
        vertical_fit, _ = fit_gaussian(vertical_x, vertical_histogram)
    except:
        horizontal_fit = [0] * len(horizontal_x)
        vertical_fit = [0] * len(vertical_x)
        
    fit_2d = fit_2d_gaussian(image)

    fig = plt.figure(figsize=(8, 8))
    thickness = 1
    thickness_1 = 0.8
    # Vertical fit
    plt.subplot(2, 2, 1)
    plt.plot(vertical_histogram, label='Data', color='blue', linewidth=thickness)
    plt.plot(vertical_x, vertical_fit, label='Gaussian fit (LSE)', color='red', linewidth=thickness)
    # Highlight the mean
    plt.axvline(v_mu, color='r', linestyle='--', label='Mean ($\mu$)', linewidth=thickness_1)
    # Illustrate sigma intervals
    if v_mu - v_sigma > 0:
        plt.axvline(v_mu - v_sigma, color='g', linestyle='--', label='$\mu - \sigma$', linewidth=thickness_1)
        plt.axvline(v_mu + v_sigma, color='g', linestyle='--', label='$\mu + \sigma$', linewidth=thickness_1)
    plt.title('X Histogram')
    plt.xlabel('X-coordinate')
    plt.ylabel('Pixel Count')
    plt.legend(loc='upper right', fontsize='small')
    # original image
    plt.subplot(2, 2, 3)
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.scatter(v_mu, h_mu, color='red', label='1D_Gaussion_fit', s=3)
    plt.scatter(fit_2d['horizontal_centroid'], fit_2d['vertical_centroid'], color='Yellow', label='2D_Gaussion_fit', s=3)
    plt.legend(loc='upper right', fontsize='small')
    # Horizontal fit
    plt.subplot(2, 2, 4)
    plt.plot(horizontal_histogram, range(len(horizontal_histogram)), label='Data', color='blue', linewidth=thickness)
    plt.plot(horizontal_fit, horizontal_x, label='Gaussian fit (LSE)', color='red', linewidth=thickness)
    plt.axhline(len(horizontal_x) - h_mu, color='r', linestyle='--', label='Mean ($\mu$)', linewidth=thickness_1)
    if len(horizontal_x) - h_mu - h_sigma > 0:
        plt.axhline(len(horizontal_x) - h_mu - h_sigma, color='g', linestyle='--', label='$\mu - \sigma$', linewidth=thickness_1)
        plt.axhline(len(horizontal_x) - h_mu + h_sigma, color='g', linestyle='--', label='$\mu + \sigma$', linewidth=thickness_1)
    plt.title('Y Histogram')
    plt.xlabel('Pixel Count')
    plt.ylabel('Y-coordinate')
    plt.legend(loc='upper right', fontsize='small')
    plt.gca().set_yticklabels(reversed(plt.gca().get_yticklabels()))   # Reverse y-axis

    plt.tight_layout()
    # plt.show()
    return fig
    




# ------------------- Other functions (remains to be use) -------------------

# def calculate_center_of_mass(image):
#     """
#     Calculate the center of mass for the white shape in a grayscale image.
#     Parameters:
#     - image: A 2D numpy array representing the grayscale image.
#     Returns:
#     - (center_x, center_y): A tuple representing the x and y coordinates of the center of mass.
#     """
#     # Ensure the image is a numpy array.
#     image = np.array(image)
#     # Calculate the total mass (sum of all pixel values).
#     total_mass = np.sum(image)
#     # Create a grid of x and y coordinates.
#     y, x = np.indices(image.shape)
#     # Calculate the weighted sum of the coordinates.
#     center_x = np.sum(x * image) / total_mass
#     center_y = np.sum(y * image) / total_mass
#     return center_x, center_y


# def remove_white_spots_morph(image, threshold=200, kernel_size=3):
#     """
#     Removes white spots from a grayscale image using morphological opening.
#     Parameters:
#     - image: A 2D numpy array representing a grayscale image.
#     - threshold: An integer brightness threshold. Pixels brighter than this will be considered.
#     - kernel_size: Size of the morphological kernel.
#     Returns:
#     - A 2D numpy array of the image after white spot noise reduction.
#     """
#     # Threshold the image to create a binary image
#     _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
#     # Create a morphological kernel
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     # Apply morphological opening
#     opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
#     # Apply mask to the original image
#     image_filtered = np.where(opening_image == 255, image, 0)
#     return image_filtered

