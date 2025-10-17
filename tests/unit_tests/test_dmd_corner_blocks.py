
from conftest import *
from ALP4 import *
import time
import cv2

# ============================
# Simple DMD Corner Blocks Test
# ============================
# This script displays a static corner blocks pattern on the DMD
# with all four corners (mode 0).

# Configuration
conf = {
    'dmd_dim': 1024,        # DMD working square area resolution
    'dmd_rotation': 38,     # DMD rotation angle for image orientation correction
}

# Initialize DMD
DMD = dmd.ViALUXDMD(ALP4(version='4.3'))

# Initialize CornerBlocksCalibrator with mode 0 (all four corners)
calibrator = simulation.CornerBlocksCalibrator(size=256, block_size=32, intensity=255, special=0)

print("Starting corner blocks display (mode 0: all four corners)...")
print("Press Ctrl+C to stop")

try:
    for i in range(1000000):
        # Generate the corner blocks pattern
        calibrator.generate_blocks()
        img = calibrator.canvas
        
        # Scale up the image to DMD resolution using macro pixels
        img = simulation.macro_pixel(img, size=int(conf['dmd_dim'] / img.shape[0]))
        
        # Apply rotation adjustment for DMD alignment
        img = dmd.dmd_img_adjustment(img, conf['dmd_dim'], angle=conf['dmd_rotation'])
        
        # Display on DMD
        DMD.display_image(img)
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping program...")

finally:
    # Clean up
    DMD.end()
    print("DMD cleaned up successfully")
