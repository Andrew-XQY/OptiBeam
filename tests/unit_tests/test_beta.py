from conftest import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

c = 0.15  # This is the Central value (mode) of the distribution
decay_factor_a = 5 # Multiplying by a factor, e.g., 5, for sharper peaks
decay_factor_b = 15
a = decay_factor_a * 2 * c  
b = decay_factor_b * 2 * (1 - c)
loc = 0.01  # This shifts the start of the range to +loc
scale = 1-loc  # This scales the distribution to span from loc to 1 - loc

# Generating a single sample
sample = []
for i in range(100000):
    sample.append(beta.rvs(a, b, loc=loc, scale=scale))

# Plotting the histogram
plt.hist(sample, bins=1000, alpha=0.75, color='blue')
plt.title('Histogram of Beta Distribution Samples')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()


