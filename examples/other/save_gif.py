import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip


def mp4_to_gif(video_path, save_path, fps=None):
    """
    Convert an MP4 video to a GIF, keeping original colors.
    
    Args:
        video_path: Path to the input MP4 file
        save_path: Path to save the output GIF
        fps: Frame rate for the GIF (None to use original video fps)
    """
    clip = VideoFileClip(video_path, audio=False)
    if fps is not None:
        clip = clip.set_fps(fps)
    clip.write_gif(save_path)
    clip.close()


def mp4_to_image_arrays(video_path):
    """
    Parse an MP4 video file into a list of numpy arrays (keeping original RGB colors).
    """
    clip = VideoFileClip(video_path, audio=False)
    image_arrays = []
    
    for frame in clip.iter_frames():
        image_arrays.append(frame)
    
    frame_rate = clip.fps
    clip.close()
    
    return image_arrays, frame_rate


if __name__ == "__main__":
    video_path = "/Users/andrewxu/Desktop/multimode fiber.mp4"
    save_path = "/Users/andrewxu/Desktop/multimode_fiber.gif"
    
    print("Loading video...")
    image_arrays, frame_rate = mp4_to_image_arrays(video_path)
    print(f"Loaded {len(image_arrays)} frames at {frame_rate} fps")
    
    # Adjust playback speed: higher fps = faster playback
    # Original is 120 fps, try 30-60 for faster GIF playback
    gif_fps = 60
    
    print(f"Creating GIF at {gif_fps} fps...")
    clip = ImageSequenceClip(image_arrays, fps=gif_fps)
    clip.write_gif(save_path)
    print(f"GIF saved to: {save_path}")