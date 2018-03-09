import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

##############################################################

# fetch and load data

# import and load numbers
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

# height
print(X_train.shape[1])

# width
print(X_train.shape[2])


def image(index):
    plt.imshow(X_train[index], cmap=plt.get_cmap('gray'))
    plt.show()
    print(y_train[index])

# image at 22th
# image(22)


# flattening the 28*28 images into 784 vector
pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], pixels).astype('float32')

# normalize the inputs from 0-255
X_train = X_train / 255
X_test = X_test / 255

# one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
classes = y_test.shape[1]

##############################################################

##############################################################

# build the model

# creating NN


def create_model():
    model = Sequential()
    model.add(Dense(pixels, input_dim=pixels,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(classes, kernel_initializer='normal', activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

##############################################################

##############################################################

# running Model


# build the model
model = create_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=100, batch_size=200, verbose=2)

# final evoluation of model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Final Error Score is : %.2f%%" % (100-scores[1]*100))

##############################################################

##############################################################

#test the model

#TEST ON A SINGLE IMAGE
#To view an MNIST image load the database into another set of vars
(X1_train, y1_train), (X1_test, y1_test) = mnist.load_data()
#to view a particular image
plt.imshow(X1_train[129], cmap=plt.get_cmap('gray'))
plt.show()
#or just get the probabilities but use 784 bit vector form
pr = model.predict(X_train[129].reshape((1, 1, 28, 28)))

print(pr)

##############################################################