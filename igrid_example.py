import cv2
import numpy as np
from disputils.igrid import Igrid

# boolean if True save to file else display
saveToFile=False

img = cv2.imread('jemma.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = img[100:300,100:312]
print("shape of image {}".format(img.shape))

# dim specified the stacking width = height as the grid is a square
dim=5

# create the grid object
ig=Igrid(dim,dim)

# stuff each cell with an image
for i in range(dim):
    for j in range(dim):
        ig.set(i,j,img)

if saveToFile:
    # save to file
    ig.save('firstgrid.png')
else:
    # else show on screen
    ig.show()
    input("Hit <enter> to continue...")
    ig.close()


# display without the image but only grids
ig1=Igrid(dim,dim)
for i in range(dim):
    for j in range(dim):
        ig1.set(i,j,img,draw=False)

if saveToFile:
    ig1.save("secondgrid.png")
else:
    ig1.show()
    input("Hit <enter> to continue...")
    ig1.close()
