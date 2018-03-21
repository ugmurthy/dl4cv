from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True, help="path to input dataset")
ap.add_argument("-o","--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b","--batch-size", type=int, default=32,
    help="batch size of the images to be passed through nextwork")
ap.add_argument("-s","--buffer-size", type=int, default=1000,
    help="size of feature extraction bufffer")
args=vars(ap.parse_args())

# store batchsize in a vars
bs = args["batch_size"]

# grab list of images that we'll be describing then randomly
# shuffle them to all for easy training and testing splits via
# array slicing during training time

print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

#extract class label from image paths
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network")
model = VGG16(weights="imagenet", include_top=False)

# initialise the HDF5 dataset writer, then store the class labels
# names in the Dataset
dataset = HDF5DatasetWriter( (len(imagePaths), 512*7*7),
    args["output"], dataKey="features", bufSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# initialise progressbar
widgets = ["Extracting Features ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ",progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
    widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialise the
    # list of actual images that will be passed throught the network
    # for feature extraction
    batchPaths = imagePaths[i:i+bs]
    batchLabels = labels[i:i+bs]
    batchImages = []

    # loop over images and labels in the current batch
    for (j,imagePath) in enumerate(batchPaths):
        # load input image using keras help func while
        # ensuring image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224,224))
        image = img_to_array(image)

        # preproccess the image by 1) expanding the dimensions random
        # 2) subtracting the mean RGB pixel intensity from the
        # imageNet Dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        #add the image to the batch
        batchImages.append(image)

    # pass the image through the network and use the outputs as
    # as our actual Features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the MaxPooling2D outputs
    features = features.reshape((features.shape[0], 512*7*7))

    # add features and labels to our HDF5 dataset
    dataset.add(features,batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
