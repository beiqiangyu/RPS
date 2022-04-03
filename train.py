import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import processImage

from tensorflow.python.client import device_lib

# 列出所有的本地机器设备

# X_train, y_train, X_test, y_test, val_images, val_labels = processImage.getDataSet()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split = 0.2)
train_set= train_datagen.flow_from_directory(
        "Data/train",
        target_size=(200,200),
        color_mode="grayscale",
        batch_size=32,
        class_mode='categorical',
        subset = 'training')
test_set = train_datagen.flow_from_directory(
    "Data/test",
    target_size = (200,200),
    color_mode="grayscale",
    batch_size=32,
    class_mode = 'categorical',
    subset = 'validation'
)
print(type(test_set))

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=(200,200,1)),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=3,activation="softmax")
])
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=["accuracy"])
Val_ACCURACY_THRESHOLD = 0.95
ACCURACY_THRESHOLD = 0.98

class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > Val_ACCURACY_THRESHOLD):
            self.model.stop_training = True
        elif(logs.get('accuracy') > ACCURACY_THRESHOLD):
            self.model.stop_training = True
callbacks = myCallback()

model.fit(
    train_set,
    # X_train,
    epochs = 20,
    validation_data = test_set,
    # validation_data = y_train,
    callbacks=[callbacks]
    )

model.save_weights('skin_v3_simple_dataset_model.h5')

with open("skin_v3_simple_dataset_model.json", "w") as json_file:
    json_file.write(model.to_json())
