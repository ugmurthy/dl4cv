# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap=argparse.ArgumentParser()
ap.add_argument("-d","--db", required=True,
    help="path to HDF5 database")
ap.add_argument("-m","--model",required=True,
    help="path to output model")
ap.add_argument("-j","--jobs",type=int, default=-1,
    help="# number of jobs to run while tuning hyperparameters")
args=vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(args["db"],"r")
# 75% training data, 25% testing
i = int(db["labels"].shape[0]*0.75)

#define a set of parameters that we want to tune then start a
# grid search where we evalaute our model for each value of C
params = {"C":[0.1,1.0,10.0,100.0,1000.0,10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])

model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# evaluate model
print("[INFO] evaluating...")
preds=model.predict(db["features"][i:])
print(classification_report(db["labels"][i:],preds, target_names=db["label_names"]))

# serialise the model to disk
print("[INFO] saving model")
f = open(args["model"],"wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close
