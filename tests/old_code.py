






# unified API for image reconstruction models
class Model(ABC):
    """
    Abstract class to define a unified API for image reconstruction models.
    
    This class serves as a base for different reconstruction models, potentially
    using various frameworks like TensorFlow or PyTorch.
    """
    
    @abstractmethod
    def speckle_reconstruction(self, image):
        """
        Reconstruct an image from speckle patterns.

        Args:
            image (np.array or similar): The input image which might be in various formats
            and needs conversion to numpy array if not already one.

        Returns:
            np.array: The reconstructed image as a numpy array.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)  # Convert to numpy array if not already
        return self._reconstruct(image)

    @abstractmethod
    def _reconstruct(self, image_array):
        """
        Implement the reconstruction logic specific to the model and framework.

        Args:
            image_array (np.array): The image as a numpy array to be reconstructed.

        Returns:
            np.array: The reconstructed image.
        """
        pass






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