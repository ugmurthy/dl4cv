# import necessary pacakges
from pyimagesearch.nn.conv import LeNet
from keras.utils import plot_model

# initialise LeNet and then write the network architecture
# visualisation graph to disk

model = LeNet.build(28,28,1,10)
plot_model(model, to_file="lenet.png",show_shapes=True)
