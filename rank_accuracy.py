# import necessary packages
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

# construct the argument parse and parse Arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--db", required=True,
    help="path to HDF5 database")
ap.add_argument("-m","--model",required=True,
    help="path to pre-trained model")
args=vars(ap.parse_args())

# load pre-trained model
print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"],"rb").read())

#open hdf5 database and split between training and test datasets
db = h5py.File(args["db"],"r")
i = int(db["labels"].shape[0]*0.75)

# make predictions on test set then compute rank-1, rank-5 accuracies
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1,rank5)=rank5_accuracy(preds,db["labels"][i:])

# display results
print("[INFO] rank-1: {:.2f}%".format(rank1*100))
print("[INFO] rank-5: {:.2f}%".format(rank5*100))

db.close()
