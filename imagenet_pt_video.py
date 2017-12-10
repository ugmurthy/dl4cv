# import the necessary pacakges
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse
import cv2
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,
    help= "path to input image directory")
ap.add_argument("-m","--model",type=str, default="vgg16",
    help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes inside keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # tensor flow only
    "resnet": ResNet50
}

IMAGETYPES = {
    ".jpg",".jpeg",".png"
}

# Ensure valid models are supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the MODELS dictionary")

# initialise the input image shape 224x224 pixels along with th pre-preprocessing function
# this needs a change based on which model we chose

inputShape=(224,224)
preprocess = imagenet_utils.preprocess_input
# if we are using the InceptionV3 or Xception networks, then we need to set
# the input shape to 299x299 and use a different image processing function
if args["model"] in ("inception","xception"):
    inputShape = (299,299)
    preprocess = preprocess_input

# load our network weights from disk
print ("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# get list of IMAGES
imgpath = args["image"]

if imgpath == 'video':
    # we are processing images from video camera
    # classify each frame from video camera
    camera = cv2.VideoCapture(0);
    frame_no = 0
    while (True):
        x=input("Hit return to contine 'q' to quit...")
        (grabbed, frame) = camera.read()
        frame_no += 1
        # convert BGR to RGB
        orig = imutils.resize(frame, width=600)
        frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        # convert to a PIL IMAGE
        frame_pil = Image.fromarray(frame)
        # resize to a square size
        image = frame_pil.resize(inputShape)
        image = img_to_array(image)
        # image ready for the imagenet pre-trained network

        # our input image is now represented by a numpy array of shape
        # inputShape[0], inputShape[1], however we need to expand the
        # dimension by making the shape (1, inputShape[0],inputShape[1],3)
        # so that we can pass it through the network
        image = np.expand_dims(image,axis=0)

        # preprocess the image using appropriate function based on model
        image = preprocess(image)

        # classify image
        print("[INFO] classifying image with '{}'".format(args["model"]))
        preds = model.predict(image)
        P = imagenet_utils.decode_predictions(preds)

        #loop over the predictions and display the rank-5 predictions probabilities
        for (i, (imagenetID,label,prob)) in enumerate(P[0]):
            print("{}. {}: {:.2f}%".format(i+1,label, prob*100))

            # load image, classify it and display with label

        (imagenetID, label,prob) = P[0][0]
        cv2.putText(orig, "label: {}".format(label), (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Classification", orig)
        #x=input("Hit return to contine 'q' to quit...")
        if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
