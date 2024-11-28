import numpy as np
import scipy.stats as stats
import cv2

def calculate_mean(data: np.array) -> float:
    """
    The average value of the dataset, which provides a central point around which the data values are distributed.
    
    Args:
        data: np.array
        
    Returns:
        float
    """
    return np.mean(data)

def calculate_median(data: np.array) -> float:
    """
    The middle value in the dataset when it is ordered from least to greatest, which can be a better measure of central tendency than the mean in skewed distributions.
    
    Args:
        data: np.array
        
    Returns:
        float
    """
    return np.median(data)

def calculate_variance(data: np.array) -> float:
    """
    A measure of the dispersion of the dataset, indicating how far each data point in the set is from the mean. Higher variance indicates more spread out data.
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return np.var(data, ddof=1)  # ddof=1 for sample variance

def calculate_standard_deviation(data: np.array) -> float:
    """
    The square root of the variance, providing a measure of dispersion in the same units as the data. It shows how much variation or dispersion exists from the average (mean).
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return np.std(data, ddof=1)  # ddof=1 for sample standard deviation

def calculate_range(data: np.array) -> float:
    """
    The difference between the maximum and minimum values in the dataset, showing the span of data points.
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return np.max(data) - np.min(data)

def calculate_iqr(data: np.array) -> float:
    """
    The range between the first quartile (25th percentile) and the third quartile (75th percentile) in the dataset. It measures the middle 50% of the data and is less sensitive to outliers.
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return np.subtract(*np.percentile(data, [75, 25]))

def calculate_kurtosis(data: np.array) -> float:
    """
    Measures the "tailedness" of the distribution. High kurtosis indicates a distribution with heavy tails and a sharp peak, while low kurtosis indicates a flatter distribution with light tails.
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return stats.kurtosis(data)  # By default, this function calculates excess kurtosis.

def calculate_skewness(data: np.array) -> float:
    """
    This is a measure that quantifies how asymmetrical the distribution is around its mean. A skewness of 0 indicates a symmetrical distribution, 
    a positive skewness indicates a tail on the right side (or more high-value outliers), and a negative skewness indicates a tail on the left side 
    (or more low-value outliers)
    
    Args:
        data: np.array
        
    Returns:
        float
    """   
    return stats.skew(data)

def get_statistics(data):
    """
    Calculate various statistical measures for a given dataset.
    
    Args:
        data: np.array
        
    Returns:
        dict
    """
    return {
        "mean": calculate_mean(data),
        "median": calculate_median(data),
        "variance": calculate_variance(data),
        "standard_deviation": calculate_standard_deviation(data),
        "range": calculate_range(data),
        "interquartile_range": calculate_iqr(data),
        "kurtosis": calculate_kurtosis(data),
        "skewness": calculate_skewness(data)
    }
    
    
# -------------------- Image Analysis --------------------
def analyze_image(image: np.array) -> dict:
    if image is None:
        return "Image data is not valid."
    # Calculate max, min, and average intensity
    max_intensity = np.max(image)
    min_intensity = np.min(image)
    average_intensity = np.mean(image)
    # Calculate standard deviation
    std_deviation = np.std(image)
    # Estimating noise level using the Laplacian operator
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    noise_estimate = np.sqrt(laplacian_var)
    return {
        f"max_intensity": max_intensity,
        f"min_intensity": min_intensity,
        f"avg_intensity": average_intensity,
        f"std": std_deviation,
        f"noise_level_estimated": noise_estimate
    }