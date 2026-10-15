"""Project local images on the DMD, one image at a time."""

from conftest import *
from ALP4 import ALP4
from pathlib import Path
import time
import cv2


conf = {
    'dmd_dim': 1024,  # DMD working square area resolution
    'dmd_rotation': -45,  # DMD rotation angle for image orientation correction
    'display_time': 2.0,  # Seconds to display each image
}

# Image folders (absolute paths or paths relative to this script).
path_to_images = [
    r'C:\Users\qiyuanxu\Documents\DataHub\local_images\MMF\procIMGs\processed',
    r'C:\Users\qiyuanxu\Documents\DataHub\local_images\MMF\procIMGs_2\processed',
]


if __name__ == '__main__':
    folders = [str(Path(__file__).resolve().parent / folder) for folder in path_to_images]
    paths = sorted(utils.get_all_file_paths(folders))
    if not paths:
        raise SystemExit('No files found. Check path_to_images above.')

    DMD = dmd.ViALUXDMD_V2(ALP4(version='4.3'))
    try:
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Skipping unreadable image: {path}')
                continue
            # Keep only the left half of the original image.
            img = img[:, :img.shape[1] // 2]
            img = cv2.resize(img, (conf['dmd_dim'], conf['dmd_dim']),
                             interpolation=cv2.INTER_NEAREST)
            img = dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation'])
            # Scale the brightest pixel to 255 while preserving grayscale detail.
            peak = int(img.max())
            if peak > 0:
                img = cv2.convertScaleAbs(img, alpha=255.0 / peak)
            print(f'Projecting: {path}')
            DMD.display_image(img)
            time.sleep(conf['display_time'])
    finally:
        DMD.end()
