from .utils import *
from .training import Model
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import norm
from skimage.metrics import structural_similarity 
from skimage import measure
from abc import ABC, abstractmethod



# ------------------- Transverse beam distribution reconstructino evaluations -------------------

def horizontal_histogram(image_array: np.array) -> np.array:
    """
    Calculate the horizontal histogram of a single-channel image.
    """
    if len(image_array.shape) != 2:
        raise ValueError("Input image array must be a 2D array for a single-channel image.")
    histogram = np.sum(image_array, axis=1)
    return histogram


def vertical_histogram(image_array: np.array) -> np.array:
    """
    Calculate the vertical histogram of a single-channel image.
    """
    if len(image_array.shape) != 2:
        raise ValueError("Input image array must be a 2D array for a single-channel image.")
    histogram = np.sum(image_array, axis=0)
    return histogram


def fit_1d_gaussian(data: np.array) -> tuple:
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    x = np.arange(len(data))
    peak = np.argmax(data)
    initial_guess = [x[peak], np.std(data), data[peak]]
    max_evaluations = 500 + 2 * len(data)

    try:
        params, _ = curve_fit(gaussian, x, data, p0=initial_guess, maxfev=max_evaluations)
        mu, sigma, amplitude = params
        fitted_gaussian = gaussian(x, mu, sigma, amplitude)
        return mu, sigma, fitted_gaussian
    except Exception as e:
        print(f"Failed to fit Gaussian: {e}")
        return None, None, None  # Provide more information on the failure


def normalize_value_base_image_dim(value: float, dim: float) -> float:
    """
    Normalize a value based on the dimensions of an image. for example, remap a value from [0,255] to [-1,1]
    e.g. value / len(image) * 2 - 1
    """
    return (value / dim) * 2 - 1


def compute_percentage_mask(image: np.array, percentage: int=95) -> tuple:
    """calculate the percentile intensity"""
    # Normalize and sort the pixel intensities 
    sorted_intensities = np.sort(image.ravel())[::-1]
    cumulative_sum = np.cumsum(sorted_intensities)
    total_intensity = cumulative_sum[-1]
    # Find the threshold for the desired percentage contour
    cutoff_index = np.where(cumulative_sum >= percentage / 100.0 * total_intensity)[0][0]
    threshold_intensity = sorted_intensities[cutoff_index]
    # Create a binary mask for the contour
    mask = image >= threshold_intensity
    return mask, threshold_intensity


def find_contours_from_binary_mask(mask: np.array) -> list:
    # Find contours at a constant value of 0.5
    contours = measure.find_contours(mask, level=0.5)
    return contours


def filtering_contours_based_on_area(contours: list, min_area: int=10) -> list: 
    """Filter out contours based on their area"""
    return [c for c in contours if cv2.contourArea(np.array(c, dtype=np.float32)) > min_area]


def calculate_total_contours_area(contours: list) -> float:
    # Calculate the area for each contour and sum them
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    return total_area


def get_transverse_beam_parameters(image: np.array) -> dict:
    mu1, std1, _ = fit_1d_gaussian(horizontal_histogram(image))
    mu2, std2, _ = fit_1d_gaussian(vertical_histogram(image))
    if all(x is not None for x in (mu1, std1, mu2, std2)):
        if all(0 <= y < image.shape[0] for y in (mu1, std1)) and all(0 <= z < image.shape[1] for z in (mu2, std2)):
            return {'horizontal_centroid': mu1, 'vertical_centroid': mu2,
                    'horizontal_width': std1, 'vertical_width': std2}
    return None


def normalize_transverse_beam_parameters(params: dict, image: np.array) -> dict:
    pass      


def transverse_beam_parameters(image: np.array) -> dict:
    """
    this function is used to calculate the beam parameters from the beam image.
    have to check the range validity of the beam parameters. and make sure the fit iteration times are limited.
    """
    pass


def analyze_image_pixel_values(image: np.array) -> dict:
    """
    Analyze pixel values in a given image represented as a numpy array.

    Args:
        image_array (np.array): A numpy array representing the image.

    Returns:
        dict: A dictionary containing the 'max', 'average', and 'min' pixel values.
    """
    max_pixel = np.max(image)
    average_pixel = np.mean(image)
    min_pixel = np.min(image)
    return {'max': max_pixel, 'average': average_pixel, 'min': min_pixel}


def get_beam_image_properties(image: np.array) -> dict:
    """return a dictionary of beam image properties depending on pixel values, beam parameters"""
    pass


def beam_image_reconstruction(trained_model: Model, test_set_sample: np.array):
    """
    focuses on applying trained model to reconstruct a testset image. 
    """
    pass




# ------------------- model training result evaluations -------------------

def read_pkl_to_dataframe(filepath):
    """
    reads a pickle file and converts it into a pandas DataFrame.
    """
    try:
        data = pd.read_pickle(filepath)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    
def training_report_tf(filepath):
    """
    Reads a pickle file containing training history data and plots the metrics.
    Return the read file as a dataframe table
    """
    try:
        # Load the data from the pickle file
        data = pd.read_pickle(filepath)
        
        # Check if data is a dictionary and suitable for conversion to DataFrame
        if not isinstance(data, dict):
            raise ValueError("Data is not a dictionary with expected format.")

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)
        
        # Plotting each column
        for column in df.columns:
            plt.plot(df[column], label=column)
        
        # Adding title and labels
        plt.title('Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()
        return read_pkl_to_dataframe(filepath)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise



# ------------------- Unified evaluation framework -------------------
class Model:
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    @abstractmethod
    def inference(self, image) -> dict:
        """Reconstruct the beam parameters using the model."""
        pass

    @abstractmethod
    def reconstruction(self, image) -> np.array:
        """Reconstruct the image using the model."""
        pass


# ------------------- Image based similarity evaluation -------------------

def ssim(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    # Determine the data range based on the image type
    if image1.dtype == 'float32' or image1.dtype == 'float64':
        data_range = image1.max() - image1.min()  # Assumes the images are normalized in the same way
    else:
        data_range = 255  # Assumes images are 8-bit unsigned integers
    # Compute SSIM between two images
    ssim_value = structural_similarity(image1, image2, data_range=data_range)
    return ssim_value


def pcc(image1, image2):
    # Flatten the images to 1D arrays
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    # Compute Pearson correlation coefficient
    correlation, _ = pearsonr(image1_flat, image2_flat)
    return correlation


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Means the two images are identical
    max_pixel = 1.0  # Assuming the image pixel values are in the range [0,1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def rmse(image1, image2):
    """
    Calculate the root mean squared error between two single-channel images.
    
    Parameters:
    image1 (np.array): First image array.
    image2 (np.array): Second image array.
    
    Returns:
    float: Root mean squared error between the two images.
    """
    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same dimensions")
    
    # Calculate RMSE
    mse = np.mean((image1 - image2) ** 2)
    return np.sqrt(mse)


# ------------------- Image to Parameters Metrics (old)  -------------------
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
    params, covariance = curve_fit(gaussian, x, y, p0=initial_guesses,
                                   maxfev=1000  # limit the number of function evaluations
                                   )
    y_fit = gaussian(x, *params)
    return y_fit, params


def center_of_mass(narray):
    total_mass = np.sum(narray)
    # Compute the x and y coordinates. We use np.arange to create arrays of x and y indexes, 
    # and np.sum to sum up the coordinates weighted by their intensity values (mass)
    y_indices, x_indices = np.indices(narray.shape)  # Get the indices for each dimension
    cx = np.sum(x_indices * narray) / total_mass
    cy = np.sum(y_indices * narray) / total_mass
    print(f"The centroid of mass is at ({cx:.2f}, {cy:.2f})")
    return cx, cy


def beam_params(narray, normalize=True):
    """
    Input: single channel narray image -> beam parameters (beam centroids, beam widths), normalized. Two 1D Gaussian fits are used.
    Output: dictionary object containing the beam parameters.
    """
    res = {"horizontal_centroid" : 0, "vertical_centroid" : 0, 
           "horizontal_width" : 0, "vertical_width" : 0}
    horizontal_x = np.arange(len(narray[0])) # x-axis (horizontal)
    vertical_x = np.arange(len(narray)) # y-axis (vertical)
    horizontal_hist = np.sum(narray, axis=0)
    vertical_hist = np.sum(narray, axis=1)
    # need to subtract the minimum value from the histogram for stability
    horizontal_hist = subtract_minimum(horizontal_hist)
    vertical_hist = subtract_minimum(vertical_hist)
    # two 1D Gaussian fits
    try:
        _, h_params = fit_gaussian(horizontal_x, horizontal_hist) # assume it is a Gaussian beam                        
        _, v_params = fit_gaussian(vertical_x, vertical_hist)
        res = {"horizontal_centroid" : h_params[1], "vertical_centroid" : v_params[1],
            "horizontal_width" : abs(h_params[2]), "vertical_width" : abs(v_params[2])}
    except: # if the fitting fails, return zero centroids, zero sigma as an indicator
        res = fit_2d_gaussian(narray)
    if normalize:
        res["horizontal_centroid"] = res["horizontal_centroid"] / len(horizontal_x)
        res["vertical_centroid"] = res["vertical_centroid"] / len(vertical_x)
        res["horizontal_width"] = res["horizontal_width"] / len(horizontal_x)
        res["vertical_width"] = res["vertical_width"] / len(vertical_x)
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
        popt, pcov = curve_fit(gaussian_2d, xy_flat, image_flat, p0=initial_guess,
                               maxfev=1000  # limit the number of function evaluations
                               )
        res = {"horizontal_centroid" : popt[1], "vertical_centroid" : popt[2],
                "horizontal_width" : popt[3], "vertical_width" : popt[4]}
    except: # if the fitting fails, return the center of the image, zero sigma can be seen as a indicator
        res = {"vertical_centroid" : 0, "horizontal_centroid" : 0,
            "vertical_width" : 0, "horizontal_width" : 0}
    return res  # Returns the optimized parameters (A, x0, y0, sigma_x, sigma_y)


# ------------------- evaluation functions -------------------
def calculate_rmse(y_actual: Iterable, y_predicted: Iterable):
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.

    Args:
    y_actual (iterable): Iterable (like a list or numpy array) of actual values.
    y_predicted (iterable): Iterable (like a list or numpy array) of predicted values.

    Returns:
    float: The RMSE of the predictions.
    """
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    mse = np.mean((y_actual - y_predicted) ** 2)  # Mean squared error
    rmse = np.sqrt(mse)  # Root Mean Square Error
    return rmse

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

    params = beam_params(image, normalize=False)
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
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(vertical_histogram, label='Data', color='blue', linewidth=thickness)
    plt.plot(vertical_x, vertical_fit, label='Gaussian fit (LSE)', color='red', linewidth=thickness)
    # Highlight the mean
    plt.axvline(v_mu, color='r', linestyle='--', label='Mean ($\mu$)', linewidth=thickness_1)
    # Illustrate sigma intervals
    if v_mu - v_sigma > 0:
        plt.axvline(v_mu - v_sigma, color='g', linestyle='--', label='$\mu - \sigma$', linewidth=thickness_1)
        plt.axvline(v_mu + v_sigma, color='g', linestyle='--', label='$\mu + \sigma$', linewidth=thickness_1)
    plt.title('Horizontal Histogram')
    plt.xlabel('Horizontal-coordinate')
    plt.ylabel('Pixel Count')
    plt.legend(loc='upper right', fontsize='small')
    # Normalize axis
    ax1.set_xticks(np.linspace(0, len(vertical_histogram)-1, 5))
    ax1.set_xticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax1.set_xlim(0, len(vertical_histogram) - 1)
    # original image
    ax2 = plt.subplot(2, 2, 3)
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.scatter(v_mu, h_mu, color='red', label='1D_Gaussion_fit', s=3)
    plt.scatter(fit_2d['horizontal_centroid'], fit_2d['vertical_centroid'], color='Yellow', label='2D_Gaussion_fit', s=3)
    plt.legend(loc='upper right', fontsize='small')
    # Normalize axis for image
    ax2.set_xticks(np.linspace(0, image.shape[1]-1, 5))
    ax2.set_xticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax2.set_yticks(np.linspace(0, image.shape[0]-1, 5))
    ax2.set_yticklabels(np.round(np.linspace(1, 0, 5), 2))  # Reversed order for y-ticks
    ax2.set_xlim(0, image.shape[1] - 1)
    ax2.set_ylim(image.shape[0] - 1, 0)  # Set to invert y-axis
    # Horizontal fit
    ax3 = plt.subplot(2, 2, 4)
    plt.plot(horizontal_histogram, range(len(horizontal_histogram)), label='Data', color='blue', linewidth=thickness)
    plt.plot(horizontal_fit, horizontal_x, label='Gaussian fit (LSE)', color='red', linewidth=thickness)
    plt.axhline(len(horizontal_x) - h_mu, color='r', linestyle='--', label='Mean ($\mu$)', linewidth=thickness_1)
    if len(horizontal_x) - h_mu - h_sigma > 0:
        plt.axhline(len(horizontal_x) - h_mu - h_sigma, color='g', linestyle='--', label='$\mu - \sigma$', linewidth=thickness_1)
        plt.axhline(len(horizontal_x) - h_mu + h_sigma, color='g', linestyle='--', label='$\mu + \sigma$', linewidth=thickness_1)
    plt.title('Vertical Histogram')
    plt.xlabel('Pixel Count')
    plt.ylabel('Vertical-coordinate')
    plt.legend(loc='upper right', fontsize='small')
    # Normalize axis
    ax3.set_yticks(np.linspace(0, len(horizontal_histogram)-1, 5))
    ax3.set_yticklabels(np.round(np.linspace(0, 1, 5), 2))
    ax3.set_ylim(0, len(horizontal_histogram) - 1)
    plt.gca().set_yticklabels(reversed(plt.gca().get_yticklabels()))   # Reverse y-axis
    plt.tight_layout()
    # plt.show()
    return fig
