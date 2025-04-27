import cv2
import numpy as np
from PIL import Image

def extract_green_border_mask(image_path: str) -> np.ndarray:
    # Load image in RGB, then convert to HSV
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green range (tweak if needed)
    lower_green = np.array([50, 100, 100])   # lower bound for H, S, V
    upper_green = np.array([90, 255, 255])   # upper bound for H, S, V

    # Create mask where green is 255 and rest is 0
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)


    return mask  # dtype=uint8, values 0 or 255

mask = extract_green_border_mask("../images/real_clouds/BlueMarbleASEAN_20250412_0000.jpg")
Image.fromarray(mask).show()

# Save the mask to a file
mask_path = "../images/mask.png"
cv2.imwrite(mask_path, mask)