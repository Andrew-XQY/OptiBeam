from .utils import *
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity 
from skimage.morphology import remove_small_objects
from skimage import measure
from abc import ABC, abstractmethod



# ------------------- Transverse beam distribution reconstructino evaluations -------------------

def vertical_histogram(image_array: np.array) -> np.array:
    """
    Calculate the vertical histogram of a single-channel image.
    Histogram of pixel sums along Y-axis (rows)
    """
    if len(image_array.shape) != 2:
        raise ValueError("Input image array must be a 2D array for a single-channel image.")
    histogram = np.sum(image_array, axis=1)
    return histogram


def horizontal_histogram(image_array: np.array) -> np.array:
    """
    Calculate the horizontal histogram of a single-channel image.
    Histogram of pixel sums along X-axis (columns)
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


def normalize_value_base_image_dim(value: float, dim: float, range: str='01') -> float:
    """
    Normalize a value based on the dimensions of an image. for example, remap a value from [0,255] to [-1,1]
    e.g. value / len(image) * 2 - 1
    """
    if range == '01':
        return value / dim
    elif range == '-11':
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


def filter_small_areas_in_mask(mask: np.array, area_threshold: int=100):
    # Label connected regions
    labeled_mask = measure.label(mask, connectivity=2)
    # Remove small objects
    filtered_mask = remove_small_objects(labeled_mask, min_size=area_threshold)
    # Convert back to binary
    filtered_mask = filtered_mask > 0
    return filtered_mask


def calculate_total_mask_area(mask: np.array) -> float:
    return np.sum(mask)


def get_transverse_beam_parameters(image: np.array) -> dict:
    hor = subtract_minimum(horizontal_histogram(image))
    ver = subtract_minimum(vertical_histogram(image))
    mu1, std1, _ = fit_1d_gaussian(hor)
    mu2, std2, _ = fit_1d_gaussian(ver)
    if all(x is not None for x in (mu1, std1, mu2, std2)):
        if all(0 <= y < image.shape[0] for y in (mu1, std1)) and all(0 <= z < image.shape[1] for z in (mu2, std2)):
            return {'horizontal_centroid': mu1, 'vertical_centroid': mu2,
                    'horizontal_width': std1, 'vertical_width': std2}
    return None


def analyze_image_pixel_values(image: np.array, comment='') -> dict:
    """
    Analyze pixel values in a given image represented as a numpy array.

    Args:
        image_array (np.array): A numpy array representing the image.

    Returns:
        dict: A dictionary containing the 'max_pixel', 'average_pixel', and 'min_pixel' pixel values.
    """
    max_pixel = np.max(image)
    average_pixel = np.mean(image)
    min_pixel = np.min(image)
    return {f'max_pixel_{comment}': max_pixel, f'average_pixel_{comment}': average_pixel, f'min_pixel_{comment}': min_pixel}




# ------------------- generate final result (wide table) -------------------

def compare_dict_values(dict1, dict2):
    """
    Compares values of two dictionaries based on corresponding keys and calculates
    the absolute difference between the values, ensuring that both values are comparable.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary with keys from the original dictionaries appended with '_diff',
              and values that are the absolute differences of comparable values from the original dictionaries.
    """
    result = {}
    # Iterate over the keys in the first dictionary
    for key in dict1:
        if key in dict2:
            # Ensure both values are not None before comparing
            if dict1[key] and dict2[key]:
                # Calculate the absolute difference and store it with the new key name
                result[key + '_diff'] = abs(dict1[key] - dict2[key])
            else:
                # Handle cases where one or both values are None
                result[key + '_diff'] = None  # or use 'None' if you prefer to record missing comparisons
    return result


def ave_dict_values(d):
    if None in d.values():
        return None
    return sum(abs(value) for value in d.values()) / len(d.values()) 


def sum_dict_values(d):
    if None in d.values():
        return None  # Return None if any value is None
    return sum(abs(value) for value in d.values())


def batch_evaluation(test_dataset: Iterable, model: tf.keras.Model):
    """inference on the test dataset and return the results in a pandas DataFrame.

    Args:
        test_dataset (Iterable): testset image data paths
        model (tf.keras.Model): trained model (tensorflow model)

    Returns:
        pd.DataFrame: a pandas DataFrame (wide table) containing the evaluation
    """
    temp = []
    
    for path in tqdm(test_dataset):
        # preprocess the image
        img = load_image_as_narray(path)
        img = image_normalize(img)
        tar, inp = split_image(img)
        # apply the trained model to the testset and get results
        reconstruction = model.predict(np.expand_dims(np.expand_dims(inp, axis=0), axis=-1), verbose=0)        
        params_predict = get_transverse_beam_parameters(np.squeeze(reconstruction))  
        params_real = get_transverse_beam_parameters(np.squeeze(tar))  
        # calculate beam parameters on real and reconstructed images
        param_names = ["horizontal_centroid", "vertical_centroid", "horizontal_width", "vertical_width"]
        diff = {}
        flag = False # if any beam parameter is not able to calculate, set problem to be true
        if not params_predict: 
            params_predict = {key: None for key in param_names}
        else:
            params_predict = {k: normalize_value_base_image_dim(v, img.shape[0]) 
                              for k, v in params_predict.items()}
        if not params_real:
            params_real = {f"{key}_real": None for key in param_names}
            flag = True
            diff = {k:None for k in params_predict.keys()}
        else:
            params_real = {k: normalize_value_base_image_dim(v, img.shape[0]) 
                           for k, v in params_real.items()}
            diff = compare_dict_values(params_predict, params_real)
            # diff = {key: value * 100 for key, value in diff.items() if value is not None}
            params_real = {f"{key}_real": value for key, value in params_real.items()}

        percentile = 90
        mask, _ = compute_percentage_mask(tar, percentile)
        mask = filter_small_areas_in_mask(mask)
        # get all image meta data with respect to the specific image.
        comment='speckle'
        meta = {
                    **params_predict, 
                    **params_real,
                    f'{percentile}_percentile_area_real': calculate_total_mask_area(mask),
                    **analyze_image_pixel_values(inp, comment=comment),  # use fiber output for stability and consistency
                    **diff,
                    'pcc': pcc(np.squeeze(tar), np.squeeze(reconstruction)),
                    'ave_params_pred_error': ave_dict_values(diff),
                    'total_params_pred_error': sum_dict_values(diff),
                    'image_path': "/".join(path.split("/")[-4:]),  # relative path
                    'image_width': tar.shape[1],
                    'image_hight': tar.shape[0],
                    'problem': flag,
                    'check': False
               }
        temp.append(meta)
    
    # construct final table 
    df_result = pd.DataFrame(temp)
    return df_result




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
def calculate_rmse(actual: Iterable, predicted: Iterable) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between actual and predicted values.
    
    Args:
        actual (Iterable): An iterable of actual values.
        predicted (Iterable): An iterable of predicted values.

    Returns:
        float: RMSE
    """
    # Filter out pairs where either actual or predicted is NaN
    clean_data = [(act, pred) for act, pred in zip(actual, predicted) if not np.isnan(act) and not np.isnan(pred)]
    if not clean_data:  # Check if all data were NaN
        print("Error: All data pairs were removed due to NaN values.")
        return np.nan
    # Unzip the cleaned list of tuples into two lists
    actual_clean, predicted_clean = zip(*clean_data)
    # Convert lists to numpy arrays
    actual_array = np.array(actual_clean)
    predicted_array = np.array(predicted_clean)
    # Calculate RMSE
    mse = np.mean((actual_array - predicted_array) ** 2)
    return np.sqrt(mse) 
