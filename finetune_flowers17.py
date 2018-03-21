# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m","--model",required=True,
    help="path to output model")
args=vars(ap.parse_args())

# construct the image generator for augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# grab list of images and class label names from image paths
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialise image processors
aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

# load the data set from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
(data,labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/ 255.0

# partition data into train and test sets
(trainX, testX, trainY, testY)= train_test_split(data, labels,
    test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load VGG16 without the top
baseModel = VGG16(weights="imagenet",include_top=False,
    input_tensor=Input(shape=(224,224,3)))

# add new FChead with a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC nodel on top of the base model - this will
# become the actual model that will be trained
model = Model(inputs=baseModel.input, outputs=headModel)


# loop over layers to freeze them and render them untrainable
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] Compiling model ...")

opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network for a few epochs - this will enable FC layer to become
# initilised with actual learned values versus random ones
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX,trainY, batch_size=32),
    validation_data=(testX, testY), epochs=25,
    steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate network after intitialisation
print("[INFO] evaluating after initialisation...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),target_names=classNames))


# now that the FC head have  been trained / initilised ,lets
# unfreeze some conv layers and make them trainable

for layer in baseModel.layers[15:]:
    layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a very small learning rate
print("[INFO] re-compiling model..." )
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model again. this time fine-tuning both the FC and the last CONV layer
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX,trainY, batch_size=32),
    validation_data=(testX, testY), epochs=100,
    steps_per_epoch=len(trainX) // 32, verbose=1)


print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),target_names=classNames))

print("[INFO] serializing model...")
model.save(args["model"])
