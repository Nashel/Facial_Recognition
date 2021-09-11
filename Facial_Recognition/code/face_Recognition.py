import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import tensorflow as tf
import cv2
import numpy as np
#preparat per olivetti_faces
def prepare_workers(dataset,dataset_target,trainNum,testNum):
    labels=[]
    train = []
    test = []
    n = 0
    imgSize=64
    for i in dataset_target:
        if i not in labels:
            labels.append(i)

    # No need to resize because all the data comes at 64x64
    
    for i in range(0,len(dataset)):
        if (n>=10):
            n=0
        if (n<trainNum-1):
            train.append([dataset[i].tolist(),dataset_target[i]])
        else:
            test.append([dataset[i].tolist(),dataset_target[i]])
        n+= 1

    train = np.array(train)
    test = np.array(test)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Reshape
    x_train.reshape(-1, imgSize, imgSize, 1)
    y_train = np.array(y_train)

    x_test.reshape(-1, imgSize, imgSize, 1)
    y_test = np.array(y_test)

    # Only if images are grayScale
    x_train.resize(len(x_train), imgSize, imgSize, 1)
    x_test.resize(len(x_test), imgSize, imgSize, 1)
    # Data augmentation, random 
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    return x_train,y_train,x_test,y_test
def create_model(x_train,y_train,x_test,y_test,n_workers):
    imgSize=64
    learningRate = 0.0001
    epochNum = 60
    # Model
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(imgSize,imgSize,1))) # GrayScale, for color: (imgSize,imgSize,3)
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(n_workers+1, activation="softmax"))

    model.summary()

    # Adam optimizer
    opt = Adam(lr=learningRate)
    model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

    history = model.fit(x_train,y_train,epochs = epochNum , validation_data = (x_test, y_test))

    return model,history
def show_results(history):
    epochNum = 60
    # Results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochNum)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

        
