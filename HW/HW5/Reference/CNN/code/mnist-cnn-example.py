# -*- coding: utf-8 -*-
"""
Dr. Valles

MNIST dataset: 60K for training / 10K for testing
CNN-model:
    Convolution (28x28x1): 32 features maps
    MaxPool: 2x2 filter window
    Convolution (11x11x32): 64 feature maps
    MaxPool: 2x2 filter window
    Convolution (3x3x64): 64 feature maps
    Flatten layer: 3x3x64=576
    Dense 1: 64 nodes
    Dense 2: 10 output with softmax

Intall TensorFlow in console:  pip install tensorflow
    
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#------------- training tensor and normalization
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

#------------ test tensor and normalization
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

#------------- string to numerical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#-------------- Creating the CNN model ---------------------------------------
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#--------------- Configuring the model compilation ---------------------------
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#--------------- Training ---------------------------------------
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#---------------- Testing predictions ----------------------------
predictions = model.predict(test_images)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy: %f", test_acc)
print("Test loss: %f", test_loss)

print("Predictions: ", predictions)
print("Predictions shape: ", test_labels)


