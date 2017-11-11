from pyimagesearch.nn import NeuralNetwork
import numpy as np

# construct XOR data set


nn=NeuralNetwork([2,2,1],alpha=0.5)



nn.fit(X,y,epochs=20000)

#now that the network is trained let loop over XOR data
for (x,target) in zip(X,y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0

    print("[INFO]  data={}, ground-truth={}, pred={:.4f}, step={}".format(
        x, target[0],pred,step))
nn.Wprint()
