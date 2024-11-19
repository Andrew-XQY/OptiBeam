from .utils import *
from io import BytesIO
from sklearn.decomposition import PCA
from moviepy.editor import ImageSequenceClip
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import imageio




# -------------- Exploretory Data Analysis --------------

def image_batch_intensity_distribution(images : np.array):
    pass



# ------------------- Plot evaluation -------------------

def plot_prediction_comparison(real : np.array, predicted : np.array, param_name=''):
    # Assuming 'real' and 'predicted' are the lists containing the actual and predicted ages
    # Scatter Plot of Predicted vs. Actual Ages
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(real, predicted, alpha=0.6, s=3)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'k--', lw=2)  # Diagonal line
    plt.title(f'Predicted vs. Actual {param_name}')
    plt.xlabel(f'Actual {param_name}')
    plt.ylabel(f'Predicted {param_name}')
    plt.grid(True)
    
    # Residual Plot
    plt.subplot(1, 3, 2)
    residuals = predicted - real
    plt.scatter(predicted, residuals, alpha=0.6, s=3)
    plt.title(f'{param_name} Residual Plot')
    plt.xlabel(f'Predicted {param_name}')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.grid(True)
    
    # Histogram or Density Plot of Prediction Errors
    plt.subplot(1, 3, 3)
    ax = sns.histplot([i*100 for i in residuals], kde=True)
    # ax.lines[0].set_color('orange')
    plt.title('Histogram of Percentage Prediction Errors')
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.xlim(-100, 100)  # Limiting x-axis to -100% to 100% for clearer visualization
    plt.tight_layout()
    plt.show()
        
        
        
# ------------------- PCA -------------------
class visualPCA:

    def __init__(self, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.pc = None

    def fit(self, data : np.array):  # narray of images (flattened)
        self.pc = self.pca.fit_transform(data) 

    def plot_2d(self):
        plt.scatter(self.pc[:, 0], self.pc[:, 1], s=2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()

    def plot_3d(self):
        # Create a 3D scatter plot using Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=self.pc[:, 0],
            y=self.pc[:, 1],
            z=self.pc[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=self.pc[:, 2],  # Set color to the third principal component
                colorscale='Viridis',  # Color scale
                opacity=0.8
            )
        )])
        # Update the layout of the plot for better visualization
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='PC 1',
                yaxis_title='PC 2',
                zaxis_title='PC 3'
            )
        )
        fig.show()

    def plot_to_memory(self, angle : int) -> BytesIO:
        fig = plt.figure(figsize=(10, 7))  # Increase figure size for larger output
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.pc[:, 0], self.pc[:, 1], self.pc[:, 2], c=self.pc[:, 2], cmap='viridis', s=2)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.view_init(elev=20., azim=angle)  # Adjust camera angle
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)  # Specify DPI for higher resolution
        buf.seek(0)  # Seek to the start of the BytesIO buffer
        plt.close(fig)  # Close the figure to free memory
        return buf
    
    def create_gif(self, save_to : str, start_angle=0, end_angle=89, nums=60, fps=30, reverse=True):
        image_buffers = [self.plot_to_memory(a) for a in np.linspace(start_angle, end_angle, nums)]
        images = [Image.open(image_buffer) for image_buffer in image_buffers]
        if reverse:
            images = images + images[::-1]
        clips = [np.array(image) for image in images]
        clip = ImageSequenceClip(clips, fps=fps)
        clip.write_gif(save_to + '/sample.gif')



# ------------------- plot image -------------------
def plot_narray(narray_img, channel=1):    
    """
    Plot a 2D NumPy array as an image.
    Parameters:
    narray_img (np.ndarray): A 2D NumPy array to plot as an image.
    """
    if np.max(narray_img) <= 1:
        narray_img = (narray_img * 255).astype(np.uint8)
    if len(narray_img.shape) == 2:
        if channel == 1:
            plt.imshow(narray_img, cmap='gray', vmin=0, vmax=255)  # cmap='gray' sets the colormap to grayscale
        else:
            plt.imshow(narray_img)
        plt.colorbar()  # Add a color bar to show intensity scale
        plt.title('2D Array Image') 
        plt.xlabel('X-axis')  
        plt.ylabel('Y-axis') 
        plt.show()
    else:
        plt.imshow(narray_img)
        plt.axis('off')
        plt.show()


def check_intensity(img, cmap='gray'):
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    # Function to be called when the mouse is moved
    def on_move(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            # Get the pixel value of the image at the given (x, y) location
            pixel_value = img[y, x]
            # Update the figure title with pixel coordinates and value
            ax.set_title(f'Pixel ({x}, {y}): {pixel_value}')
            fig.canvas.draw_idle()
            
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.colorbar(im, ax=ax)  # Shows the color scale
    plt.show()


def save_gif(image_arrays, frame_rate, save_path):
    """
    Saves a list of numpy arrays as a GIF.

    Args:
    image_arrays (list of np.array): List of numpy arrays where each array represents an image frame.
    frame_rate (float): Number of frames per second.
    save_path (str): Path to save the GIF file.

    Returns:
    None
    """
    with imageio.get_writer(save_path, mode='I', duration=1/frame_rate) as writer:
        for img in image_arrays:
            if img.dtype != np.float64:
                img = (img * 255).astype(np.float64)  # Normalize and convert to uint8 if not already
            writer.append_data(img)
            
            
def save_as_matplotlib_style_gif(image_arrays, frame_rate, save_path, cmap='gray'):
    """
    Saves a list of numpy arrays as a GIF, styled to resemble matplotlib plots,
    while maintaining the original pixel values.

    Args:
    image_arrays (iterable of np.array): Iterable of numpy arrays where each array represents an image.
    frame_rate (float): Number of frames per second.
    save_path (str): Path to save the GIF file.

    Returns:
    None
    """
    images = []
    # Find global min and max across all images to set a fixed range for the colormap
    vmin = min([img.min() for img in image_arrays])
    vmax = max([img.max() for img in image_arrays])
    
    for img in image_arrays:
        fig, ax = plt.subplots()
        # Ensure that imshow uses the same color scale for all frames; set aspect for proper scaling
        ax.imshow(img, aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')  # Optionally turn off the axis.

        # Convert the matplotlib plot to an image array
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Append the styled image to the list for GIF creation
        images.append(plot_image)
        plt.close(fig)
        
    # Create GIF using moviepy
    clip = ImageSequenceClip(images, fps=frame_rate)
    clip.write_gif(save_path)


def create_gif_from_png_paths(png_paths, save_path, duration):
    """
    Creates a GIF from a list of .png file paths with the original image appearance.

    Args:
        png_paths (list of str): List of paths to .png files.
        save_path (str): Path where the GIF should be saved.
        duration (float): Duration of each frame in the GIF, in seconds.

    Returns:
        None
    """
    images = [imageio.imread(path) for path in png_paths]
    imageio.mimsave(save_path, images, duration=duration)


# ------------------- Fundamental evaluation plots -------------------

def plot_prediction_comparison(real : np.array, predicted : np.array, param_name='', directory=''):
    # Assuming 'real' and 'predicted' are the lists containing the actual and predicted ages
    # Scatter Plot of Predicted vs. Actual Ages
    legend_font = 10
    x1 = predicted[0]
    x2 = predicted[1]
    y1 = real[0]
    y2 = real[1]
    # Scatter Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    scatter_1 = plt.scatter(y1, x1, color='blue', label=f'Horizontal {param_name}', alpha=0.5, s=1)
    scatter_2 = plt.scatter(y2, x2, color='red', label=f'Vertical {param_name}', alpha=0.5, s=1)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'k--', lw=2, alpha=0.8)  # Diagonal line
    plt.title(f'Estimation vs. Actual {param_name}')
    plt.xlabel(f'Actual {param_name}')
    plt.ylabel(f'Predicted {param_name}')
    plt.legend(fontsize=legend_font, markerscale=3, loc='upper left')  # You can adjust the font size as needed
    plt.grid(True)
    # Residual Plot
    plt.subplot(1, 3, 2)
    residuals_1 = x1 - y1
    residuals_2 = x2 - y2
    plt.scatter(x1, residuals_1, color='blue', label=f'Horizontal {param_name}', alpha=0.5, s=1)
    plt.scatter(x2, residuals_2, color='red', label=f'Vertical {param_name}', alpha=0.5, s=1)
    plt.title(f'{param_name} Residual')
    plt.xlabel(f'Predicted {param_name}')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    plt.legend(fontsize=legend_font, markerscale=3, loc='upper left') 
    plt.grid(True)
    # Fit a Gaussian (normal) distribution to the data
    plt.subplot(1, 3, 3)    
    data = [i * 100 for i in residuals_1]
    sns.histplot(data, color='blue', label=f'Horizontal {param_name}', kde=False, stat='density',
                 bins=200, alpha=0.3)
    x = np.linspace(min(data), max(data), 100)
    gaussian_curve = norm.pdf(x, *norm.fit(data))  # Gaussian PDF
    plt.plot(x, gaussian_curve, color='blue')
    data = [i * 100 for i in residuals_2]
    sns.histplot(data, color='red', label=f'Vertical {param_name}', kde=False, stat='density',
                 bins=200, alpha=0.3)
    x = np.linspace(min(data), max(data), 100)
    gaussian_curve = norm.pdf(x, *norm.fit(data))  # Gaussian PDF
    plt.plot(x, gaussian_curve, color='red')
    plt.title('Percentage Prediction Errors')
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.legend(fontsize=legend_font, loc='upper left') 
    plt.grid(True)
    plt.tight_layout()
    # Save the file
    if directory:
        # plt.xlim(-100, 100)  # Limiting x-axis to -100% to 100% for clearer visualization
        timestamp = datetime.now().strftime("%M%S%f")[-5:]
        filename = f'beam_parameter_estimation_results_{timestamp}.png'
        full_path = os.path.join(directory, filename)
        plt.savefig(full_path, transparent=True, format='png', dpi=300)
    plt.show()