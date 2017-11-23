from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialised file, and the starting epoch
        super(TrainingMonitor,self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        # intilise the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check if starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update loss and accuracy info
        # for the entire training process
        for (k,v) in logs.items():
            l=self.H.get(k,[])
            l.append(v)
            self.H[k]=l

        # check to see if training history should be serialised to file
        if self.jsonPath is not None:
            f = open(self.jsonPath,"w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        if len(self.H["loss"])>1:
            # plot training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training loss and accuracy [Epoch {} ]".format(len(self.H["loss"])))
            plt.ylabel("Loss/Accuracy")
            plt.xlabel("Epoch #")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()
