# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO loading images...]")
imagePaths = list(paths.list_images(args["dataset"]))

sp= SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

#load the data set from disk then scale the raw picel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition data to training and test sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# init optimiser
print ("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX,testY),
    batch_size=32, epochs=100, verbose=1)

# evaluate network
print("[INFO] evaluating network...")
predictions = model.predict(testX,batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=["cat","dog","panda"]))

json_str = model.to_json()
print("Model Architecture\n{}".format(json_str))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
