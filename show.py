import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


img = mpimg.imread('jemma.png')
print(img.shape)
plt.style.use("ggplot")
plt.figure()
plt.show(img)
