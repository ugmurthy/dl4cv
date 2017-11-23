# set the matplotlib backend so that figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import necessary packages
from pyimagesearch.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

# construct the argument parse and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help = "path to the output directory")
ap.add_argument("-b","--bnorm", required=True,
    help = "yes or no to indicate BatchNormalization required or not")

args = vars(ap.parse_args())

#show information on process id
print("[INFO] process ID: {}".format(os.getpid()))

#load the training and test data, then scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY),(testX,testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialise label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile","bird","cat","deer", "dog", "frog",
    "horse", "ship", "truck"]

#initialise the optimiser model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10, bn=args["bnorm"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# construct callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
