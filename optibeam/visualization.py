from .utils import *
from io import BytesIO
from sklearn.decomposition import PCA
from moviepy.editor import ImageSequenceClip
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches
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
            plt.imshow(narray_img, cmap='gray')  # cmap='gray' sets the colormap to grayscale
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


def img_2_params_evaluation(image, true_label, pred_label):
    fig, ax = plt.subplots()
    ax.imshow(image.squeeze(), cmap='gray')  # Display the image

    # Calculate normalized coordinates based on image dimensions
    # These are used for plotting the centroids and ellipses
    true_x = true_label[0] * image.shape[1]
    true_y = true_label[1] * image.shape[0]
    pred_x = pred_label[0] * image.shape[1]
    pred_y = pred_label[1] * image.shape[0]

    # Plot centroids with more professional styling
    ax.plot(true_x, true_y, 'o', markersize=3, markeredgecolor='blue', markerfacecolor='none', label='True Centroid')
    ax.plot(pred_x, pred_y, '^', markersize=3, markeredgecolor='darkred', markerfacecolor='none', label='Predicted Centroid')

    # Plot ellipses with professional style
    true_ellipse = matplotlib.patches.Ellipse((true_x, true_y),
                                              width=true_label[2] * image.shape[1] * 2, 
                                              height=true_label[3] * image.shape[0] * 2,
                                              edgecolor='blue', facecolor='none',
                                              linewidth=1, linestyle='--', label='True Widths')
    ax.add_patch(true_ellipse)
    pred_ellipse = matplotlib.patches.Ellipse((pred_x, pred_y),
                                              width=pred_label[2] * image.shape[1] * 2,
                                              height=pred_label[3] * image.shape[0] * 2,
                                              edgecolor='darkred', facecolor='none',
                                              linewidth=1, linestyle='--', label='Predicted Widths')
    ax.add_patch(pred_ellipse)

    # Set labels and title with normalized axis labels
    ax.set_xlabel('Normalized Horizontal Position')
    ax.set_ylabel('Normalized Vertical Position')
    #ax.set_title('img2params model\'s prediction on a random testset sample', pad=20)

    # Improve the granularity of axis labels
    num_ticks = 10  # More ticks for better granularity
    tick_values = np.linspace(0, 1, num_ticks)
    tick_labels = [f"{x:.1f}" for x in tick_values]
    ax.set_xticks(tick_values * image.shape[1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_values * image.shape[0])
    ax.set_yticklabels(tick_labels)

    plt.legend()
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
            


def save_as_matplotlib_style_gif(image_arrays, frame_rate, save_path):
    """
    Saves a list of numpy arrays as a GIF, styled to resemble matplotlib plots.

    Args:
    image_arrays (iterable of np.array): Iterable of numpy arrays where each array represents an image.
    frame_rate (float): Number of frames per second.
    save_path (str): Path to save the GIF file.

    Returns:
    None
    """
    images = []
    for img in image_arrays:
        # Plot the image array with matplotlib to capture the style
        fig, ax = plt.subplots()
        ax.imshow(img, aspect='equal', cmap='viridis')  # 'viridis' is a common matplotlib colormap
        #ax.axis('off')  # Hide axes for a cleaner look

        # Convert the matplotlib plot to an image array
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Append the styled image to the GIF
        images.append(plot_image)
        plt.close(fig)
        
    clip = ImageSequenceClip(images, fps=frame_rate)
    clip.write_gif(save_path)





