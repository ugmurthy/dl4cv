# import necessary files
import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image widht, height, and interpolation
        # method used when resizing
        self.width = width
        self.height= height
        self.inter = inter

    def preprocess(self, image):
        # grab the dimensions of the image and then initialise
        # the deltas to when to use cropping
        (h,w) = image.shape[:2]
        dW = 0
        dH = 0

        # if width is smaller than height, then resize along widht
        # and then update the deltas to crop the height to desired
        # dimernsions
        if w < h:
            image = imutils.resize(image, width=self.width,
                inter = self.inter)
            dH = int((image.shape[0]-self.height) /2.0)
        else:
            image = imutils.resize(image, height=self.height,
                inter = self.inter)
            dW = int((image.shape[1]-self.width) /2.0)

        # finally, resize the image to provided spatial
        # dimensions to ensure out output image is always
        # a fixed size
        return cv2.resize(image,(self.width,self.height),
            interpolation=self.inter )
