import numpy as np
import matplotlib.pyplot as plt

def quadrupole_transform(data, focus_strength=1.0, defocus_strength=1.0):
    """
    Apply a quadrupole-like transformation to a 2D data distribution.
    Positions out of bounds are considered as 'squeezed out'.

    Parameters:
    - data: 2D numpy array representing the data distribution.
    - focus_strength: Strength of focusing effect.
    - defocus_strength: Strength of defocusing effect.

    Returns:
    - Transformed 2D numpy array.
    """
    # Get the center of the data
    center_x, center_y = data.shape[1] / 2, data.shape[0] / 2

    # Prepare the transformed data array
    transformed_data = np.zeros_like(data)

    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            # Calculate distance from the center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance == 0:
                continue  # Avoid division by zero at the center

            # Calculate scaling factors according to the inverse-square law
            focus_scale = focus_strength / distance**2
            defocus_scale = defocus_strength / distance**2

            # Apply focusing/defocusing transformations
            new_x = int(center_x + (x - center_x) * focus_scale)
            new_y = int(center_y + (y - center_y) * defocus_scale)

            # If the new position is within bounds, update it in the transformed data
            if 0 <= new_x < data.shape[1] and 0 <= new_y < data.shape[0]:
                transformed_data[new_y, new_x] += data[y, x]
            # Out-of-bounds positions are ignored (squeezed out)

    return transformed_data

# Example usage
# Generate a random 2D data distribution
data = np.random.rand(100, 100)

# Apply the quadrupole transformation
transformed_data = quadrupole_transform(data, focus_strength=5.0, defocus_strength=0.2)

# Plot the original and transformed data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='viridis')
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.imshow(transformed_data, cmap='viridis')
plt.title('Quadrupole Transformed Data')
plt.show()
