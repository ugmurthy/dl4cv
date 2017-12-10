# SAMPLE CODE to demonstrate reading a image file and converting to PIL image
# and then converting back to numpy asarray
# note the numpy array is in RGB format and not BGR as required by OpenCV


# import pacakges as needed
import cv2
import numpy as np
from PIL import Image



img = cv2.imread("imgs/dl1.png")

# You may need to convert the color.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)

# For reversing the operation:
im_np = np.asarray(im_pil)
