import numpy as np
import scipy.stats as stats

def calculate_mean(data: np.array) -> float:
    return np.mean(data)

def calculate_median(data: np.array) -> float:
    return np.median(data)

def calculate_mode(data: np.array) -> tuple:
    mode_result = stats.mode(data)
    return mode_result.mode[0], mode_result.count[0]  # Returns mode and count of mode

def calculate_variance(data: np.array) -> float:
    return np.var(data, ddof=1)  # ddof=1 for sample variance

def calculate_standard_deviation(data: np.array) -> float:
    return np.std(data, ddof=1)  # ddof=1 for sample standard deviation

def calculate_range(data: np.array) -> float:
    return np.max(data) - np.min(data)

def calculate_iqr(data: np.array) -> float:
    return np.subtract(*np.percentile(data, [75, 25]))

def calculate_kurtosis(data: np.array) -> float:
    return stats.kurtosis(data)  # By default, this function calculates excess kurtosis.

def calculate_skewness(data: np.array) -> float:
    return stats.skew(data)

