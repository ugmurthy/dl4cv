# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# construct the argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True,
    help = "path to the output loss/accuracy plot")
args= vars(ap.parse_args())

# load the training and test data, scale t and into the range [0,1]
# then reshape the design matrix
# cifar 10 is a 60000 images of size 32x32x3
print("[INFO] loading CIFAR-10 data...")
# trainX is (50000,32,32,3) testX is (10000,32,32,3)
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0],3072))
testX = testX.reshape((testX.shape[0],3072))
# trainX is now (50000,3072) and testX is (10000,3072)

# convert lables from integers to vections - one hot
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialise the label names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

#define the network as 3072-1024-512-10 using keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# train the model using SGD
print("[INFO] training network...")
sgd=SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=100, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
