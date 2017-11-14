# import necessary pacakages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    # grab the spatial dimensions of the image kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for the output image, taking care to "pad"
    # the orders of the input image so the spatial size (i.e. Width, Height) are not reduced
    pad = (kW-1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH,iW),dtype="float")

    # loop over the image, "sliding" the kernel across
    # each (x,y)-co-ordinate from left-to-right and top-to-bottom

    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            # NOTE: x,y starts from 0,0 of original image
            # extract ROI of the image by extracting the
            # *center* region of the current (x,y)-co-ordinates dimensions
            roi = image[y-pad:y+pad+1,x-pad:x+pad+1]

            # perform the actual convolution by taking the element-wise
            # multiplication and then summing it up
            k = (roi * K).sum()

            # store the convolved value in output(x,y). NOTE: output is not padded
            output[y-pad, x-pad]= k
    # rescale the output image to be in range [0,255]
    output = rescale_intensity(output,in_range=(0,255))
    output = (output * 255).astype("uint8")

    #return the output image
    return output


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help="path to input image filename")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7,7), dtype="float") * (1.0 / (7.0*7.0))
largeBlur = np.ones((21,21), dtype="float") * (1.0 / (21.0*21.0))

# sharpening filter
sharpen = np.array((
    [0,-1, 0],
    [-1,5,-1],
    [0,-1, 0]), dtype="int")

# construct the laplacian kernel to detect edges
laplacian = np.array((
    [0,1, 0],
    [1,-4,1],
    [0,1, 0]), dtype="int")

# Sobel kernel to detect edge along x-axis
sobelX = np.array((
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]), dtype="int")
# sobel Y
sobelY = np.array((
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]), dtype="int")

emboss = np.array((
    [-2,-1,0],
    [-1,1,1],
    [0,1,2]), dtype="int")

# kernelBank a list of kernels
kernelBank = (
    ("smallBlur",smallBlur),
    ("largeBlur",largeBlur),
    ("sharpen",sharpen),
    ("laplacian",laplacian),
    ("sobelX",sobelX),
    ("sobelY",sobelY),
    ("emboss", emboss))

#load the input image and covert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
for (kernelName, K) in kernelBank:
    # apply kerney using both our function convolve and OpenCVs filter2D function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray,K)
    opencvOutput = cv2.filter2D(gray,-1,K)

    #cv2.imshow("original",gray)
    #cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("1.original      2.{} - convolve        3.{} - opencv  ".format(kernelName,kernelName),
        np.hstack((gray,convolveOutput,opencvOutput)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
