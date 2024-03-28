from .utils import *
from io import BytesIO
from sklearn.decomposition import PCA
from moviepy.editor import ImageSequenceClip
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


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
        