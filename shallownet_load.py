# import the necessary packages
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m","--model", required=True,
    help="path to pre-trained model")
args = vars(ap.parse_args())

# initialise the class labels
classlabels = ["cat", "dog", "panda"]


# grab images in dataset then randomly sample indexes
# into the image paths
print("[INFO sampling images...]")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# init preprocessors
sp= SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

#load the data set from disk then scale the raw picel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

#load pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# predict
print("[INFO] predicting...")
preds = model.predict(data,batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    #load the sampled image, draw prediction and display it on screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classlabels[preds[i]]),
        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)
