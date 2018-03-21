# import necessary packages
import numpy as np

def rank5_accuracy(preds,labels):
    # initialise rank-1 and rank-5 accuracies
    rank1 = 0
    rank5 = 0

    #loops over the predictions and ground truth labels
    for (p, gt) in zip(preds, labels):
        # sort the probablities by their index in descending order
        # so that the confident guess are the the front of the List
        p = np.argsort(p)[::-1]

        #check if ground truth label is in top 5
        # predictions
        if gt in p[:5]:
            rank5 +=1

        # check to see if the ground truth is the #1 prediction
        if gt == p[0]:
            rank1 +=1

    # compute final rank-1 and rank-5 accuracies
    rank1 /= float(len(labels))
    rank5 /= float(len(labels))

    # return a tuple of rank1 and rank5 accuracies
    return(rank1, rank5)
