# import necessary pacakages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def pool(image, ops="max"):
    # grab the spatial dimensions of the image kernel
    (iH, iW) = image.shape[:2]
    # filter is a 2x2 matrix
    # allocate memory for the output image, taking care to "pad"
    # the orders of the input image so the spatial size (i.e. Width, Height) are not reduced
    pad = 0
    stride=2
    output = np.zeros((iH//2,iW//2),dtype="float")
    #print ("Sized of input iH={} , iW={}".format(iH,iW))
    #print("Size of Output image: {}".format(output.shape))
    # loop over the image, "sliding" the kernel across
    # each (x,y)-co-ordinate from left-to-right and top-to-bottom
    (ox,oy) = (0,0)
    for y in np.arange(0, iH, stride):
        for x in np.arange(0, iW, stride):
            # NOTE: x,y starts from 0,0 of original image
            # extract ROI of the image by extracting the
            # *center* region of the current (x,y)-co-ordinates dimensions
            roi = image[y:y+2,x:x+2]

            # perform the actual convolution by taking the element-wise
            # multiplication and then summing it up
            if ops == "max":
                k = np.amax(roi)
            if ops == "sum":
                k = sum(sum(roi))/1.0
            if ops == "avg":
                k = sum(sum(roi))/4.0
            if ops == "min":
                k = np.amin(roi)
            if ops == "neg":
                k = 255.0 - np.amax(roi)
            if ops == "sqr":
                k = sum(sum(roi*roi)) * 1.0

            # store the convolved value in output(x,y). NOTE: output is not padded
            output[oy,ox]= k * 1.0
            ox += 1
        oy += 1
        ox = 0
    # rescale the output image to be in range [0,255]
    #print("Before {}".format(output[10:15,10:15]))
    output = rescale_intensity(output,in_range=(0,255))
    #print("After rescale {}".format(output[10:15,10:15]))
    output = (output * 255).astype("uint8")
    #print("After recast, {}".format(output[10:15,10:15]))
    #return the output image
    return output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input image filename")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resizing to 200x200
gray = gray[100:300,100:300]

print("[INFO] applying MAX Pool")
convolveOutput = pool(gray,ops="max")
cv2.imshow("original ", gray)
cv2.imshow("Max pool", convolveOutput)
convolveOutput = pool(gray,ops="sum")
cv2.imshow("Sum pool", convolveOutput)
convolveOutput = pool(gray,ops="avg")
cv2.imshow("Avg pool", convolveOutput)
convolveOutput = pool(gray,ops="min")
cv2.imshow("Min pool", convolveOutput)
convolveOutput = pool(gray,ops="neg")
cv2.imshow("Neg pool", convolveOutput)
convolveOutput = pool(gray,ops="sqr")
cv2.imshow("Sqr pool", convolveOutput)

cv2.waitKey(0)
cv2.destroyAllWindows()
