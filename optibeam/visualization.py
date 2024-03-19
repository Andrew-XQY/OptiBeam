from .utils import *
from io import BytesIO
from sklearn.decomposition import PCA
from moviepy.editor import ImageSequenceClip
        
        
# ------------------- PCA -------------------
class visualPCA:
    def __init__(self, n_components=3):
        self.pca = PCA(n_components=n_components)
        self.pc = None

    def fit(self, data):
        self.pc = self.pca.fit_transform(data) # narray of images (flattened)

    def plot_2d(self):
        plt.scatter(self.pc[:, 0], self.pc[:, 1], s=2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()

    def plot_3d(self):
        fig = plt.figure(figsize=(10, 7))  # Increase figure size for larger output
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.pc[:, 0], self.pc[:, 1], self.pc[:, 2], c=self.pc[:, 2], cmap='viridis', s=2)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.show()

    def plot_to_memory(self, angle):
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
    
    def create_gif(self, save_To, start_angle=0, end_angle=89, nums=60, fps=30, reverse=True):
        image_buffers = [self.plot_to_memory(a) for a in np.linspace(start_angle, end_angle, nums)]
        images = [Image.open(image_buffer) for image_buffer in image_buffers]
        if reverse:
            images = images + images[::-1]
        clips = [np.array(image) for image in images]
        clip = ImageSequenceClip(clips, fps=fps)
        clip.write_gif(save_To)


