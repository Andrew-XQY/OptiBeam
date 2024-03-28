from .utils import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
        popt, pcov = curve_fit(gaussian_2d, xy_flat, image_flat, p0=initial_guess)
        res = {"horizontal_centroid" : popt[1], "vertical_centroid" : popt[2],
                "horizontal_width" : popt[3], "vertical_width" : popt[4]}
    except: # if the fitting fails, return the center of the image, zero sigma can be seen as a indicator
        res = {"vertical_centroid" : 0, "horizontal_centroid" : 0,
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
    


# ------------------- dataset clean -------------------
