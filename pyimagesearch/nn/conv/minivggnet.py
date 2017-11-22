# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes, bn="yes"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using channels_first, update the input shape
        # and chanDim
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        #
        # first CONV => RELU => CONV => RELU = POOL layer set
        model.add(Conv2D(32,(3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        if bn=="yes":
            model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        if bn=="yes":
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU = POOL layer set
        model.add(Conv2D(64,(3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        if bn=="yes":
            model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(Activation("relu"))
        if bn=="yes":
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # first and only set of FC => RELU
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        if bn=="yes":
            model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model