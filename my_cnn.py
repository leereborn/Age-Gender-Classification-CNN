from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import initializers
from keras.utils import plot_model

class CNN():
    def __init__(self, input_size):
        # Initialising the CNN
        self.classifier = Sequential()
        initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # First conv layer
        # common practice for the first CNN layer is 32 filters (the dimensionality of the output space)
        self.classifier.add(Conv2D(96, (7, 7), input_shape = (input_size, input_size, 3), activation = 'relu', kernel_initializer = initializer))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

        # Second conv layer
        self.classifier.add(Conv2D(256, (5, 5), activation = 'relu',kernel_initializer = initializer))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

        # Third conv layer
        self.classifier.add(Conv2D(384, (3, 3), activation = 'relu',kernel_initializer = initializer))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

        # Flatten layer
        self.classifier.add(Flatten())

        # Fully connected layer
        self.classifier.add(Dense(units = 512, activation = 'relu',kernel_initializer = initializer))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units = 512, activation = 'relu',kernel_initializer = initializer))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units = 1, activation = 'sigmoid',kernel_initializer = initializer))
        # classifier.add(Dense(units = 2, activation = 'softmax',kernel_initializer = initializer))

        # Compiling the CNN. Stochastic gradient process is implied here.
        # optimizer: gradient decent
        # loss: if more than two output, need to use categorical_crossentropy
        self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    def get_classifier(self):
        return self.classifier

    def plot_model(self):
        plot_model(self.classifier,to_file='model.png')