# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:31:10 2023

@author: TOM3O
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# baseline cnn model for mnist
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer

from tensorflow.keras.optimizers import SGD
import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#%%
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='leaky_relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
#
# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)
#
# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
#%%
autoencoder.compile(loss='MSE')
#%%

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
x_len = x_train.shape[1]
x_width = x_train.shape[2]
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
#%%
start = time.time()
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1,
                validation_data=(x_test, x_test))
end = time.time()
elapsed_time = end - start
print(elapsed_time)
#%%
# stores scores
# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
x_test_auto = decoder.predict(encoded_imgs)

#
# Use Matplotlib (don't ask)

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_auto[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(autoencoder.history.history['loss'], color='blue', label='train')
plt.plot(autoencoder.history.history['val_loss'], color='orange', label='test')
plt.title("Auto-Encoder Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

x_train_flat = x_train
x_train = x_train.reshape(x_train.shape[0], x_len, x_width, 1)
x_test_flat = x_test
x_test = x_test.reshape(x_test.shape[0], x_len, x_width, 1)
x_test_auto_flat = x_test_auto
x_test_auto = x_test_auto.reshape(x_test_auto.shape[0], x_len, x_width, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#%%

# example of loading the mnist dataset


# define cnn model
def define_model(mode):
    if mode == "Raw" or mode == "Auto-Encoded":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    elif mode == "Linear":
        model = Sequential()
        model.add(InputLayer(input_shape=(784, ))) # input layer
        model.add(Dense(256, activation='relu')) # hidden layer 1
        model.add(Dense(256, activation='relu')) # hidden layer 2
        model.add(Dense(10, activation='softmax')) # output layer
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    elif mode == "Small_Linear":
        model = Sequential()
        model.add(InputLayer(input_shape=(784, ))) # input layer
        model.add(Dense(10, activation='sigmoid')) # hidden layer 2
        model.add(Dense(10, activation='sigmoid')) # output layer
        model.compile(loss='MSE',
                    # optimizer='adam',
                    metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, mode, n_folds=1):
    scores = list()
    # prepare cross validation
    # enumerate splits
    #
        
    # define model
    model = define_model(mode)
    # select rows for train and test
    x_train, y_train = dataX[0:(int)(len(dataX)*0.8)], dataY[0:(int)(len(dataX)*0.8)]
    x_test, y_test = dataX[(int)(len(dataX)*0.8):], dataY[(int)(len(dataX)*0.8):]
    # fit model
    history = model.fit(x_train, y_train, epochs=5, batch_size=1, validation_data=(x_test, y_test))
    # evaluate model
    _, acc = model.evaluate(x_test, y_test)
    print('> %.3f' % (acc * 100.0))
    # stores scores
    scores.append(acc)
    
    
    return scores, history, model
 
# plot diagnostic learning curves
def summarize_diagnostics(history, mode):
    # plot accuracy
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    if mode == "Raw" or mode == "Auto-Encoded":
        plt.title("Convolutional Neural Network Training History on " + mode + " Data")
    elif mode == "Linear":
        plt.title("Multi-Layer Neural Network Training History on Latent Space Data")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
#%%

scores_dict = dict()
history_dict = dict()
# run the test harness for evaluating a model
for mode in ["Small_Linear"]:#, "Linear", "Raw", "Auto-Encoded"]:
    # load dataset
    # evaluate model
    if mode == "Raw":
        x = x_test
    elif mode == "Auto-Encoded":
        x = x_test_auto
    elif mode == "Linear":
        x = x_test_flat
    elif mode == "Small_Linear":
        x = x_test_flat
    scores, history, model = evaluate_model(x[:1000], y_test[:1000], mode)
    scores_dict[mode] = scores
    history_dict[mode] = history
    # learning curves
    summarize_diagnostics(history, mode)
    # summarize estimated performance
    
    

