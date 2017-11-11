# import the necessary packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialise the list of weight matrices, then store the network architecture
        # and learning rate [NOTE: W is a LIST of numpy ARRAYS or MATRICES]
        self.W = []
        self.alpha = alpha
        self.layers = layers
        # layer is a list of integers which represents the actual architecture
        # for eg [2,2,1] input layer of 2 nodes, followed by hidden layer of 2 nodes
        # and one output layer with 1 node

        #start looping from the index of first layer but stop before we reach last 2 layers
        for i in np.arange(0,len(layers)-2):
            # randomly initialise a weight matrix connecting the number of nodes in
            # each respective layers , adding an extra node for bias
            w = np.random.randn(layers[i]+1,layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))

        # NOTE: len(W) will be one less than len(layers)
        # the last two layers are special case where the input
        # connection need a bias term but but the output does not
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append( w / np.sqrt(layers[-2]))

    def Wprint(self):
        print("[WT Matrix] {}".format(self.W))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("->".join(str(l) for l in self.layers))

    def sigmoid(self,x):
        # computer and return sigmoid activation value of given input
        return 1.0 / (1+np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute deravative of sigmoid function ASSUMING that 'x' has already been passed
        # through sigmoid function
        return x * (1-x)

    def fit(self, X,y, epochs=1000, displayUpdate=100):
        # insert col of 1's for bias to make it a trainable parameter
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over desired epochs
        for epoch in np.arange(0,epochs):
            # loop over each data point and train our network on it
            for (x, target) in zip(X,y):
                self.fit_partial(x,target)

            # check to see if we should display a training update?
            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1,loss))

    def fit_partial(self, x,y):
        # 'x' is an individual data point from our design matrix
        # 'y' is the class label
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first activations
        # is a special case -- it's just the input feature itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by taking the
            # dot product between activation and weight matrix -- this is called
            # the "net input" to the current layer
            #print("[class] W={}".format(WW))
            net = A[layer].dot(self.W[layer])

            #computing the net output is simply applying sigmoid to net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the difference
        # between our prediction (final output activation in activations list) and
        # true target value 'y'
        error = A[-1]-y

        # from here we need to apply the chain rule and build our list of deltas 'D'
        # the first entry in the deltas is simply the error of the output layer times
        # the deravative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # loop over the layers in reverse order except the last two since we have
        # already accounted for it
        for layer in np.arange(len(A)-2,0,-1):
            # the delta for current layer is delta for previous layer
            # dot weight matrix of current layer, followed by multiplying the delta
            # by the deravative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            # .T is transpose
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over the layers in reverse order we need to set that
        # straight
        D = D[::-1]

        # WEIGHT UPDATE phase
        # lover over layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual learning takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to
		# obtain the final prediction
        p = np.atleast_2d(X)

		# check to see if the bias column should be added
        if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]

		# loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value `p`
			# and the weight matrix associated with the current layer,
			# then passing this value through a non-linear activation
			# function
            p = self.sigmoid(np.dot(p, self.W[layer]))

		# return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # make predictions for input data and then compute loss
        targets = np.atleast_2d(targets)
        predictions=self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions- targets)**2)
        return loss
