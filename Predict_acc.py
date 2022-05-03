import os.path
import random

import cv2

import processImage
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.models import model_from_json

# with open('skin_v4_6_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("skin_v4_6_model.h5")
# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

model = load_model("skin_resnet_v3_model40.h5")

def predict_acc(model, X, y):
    scores = model.evaluate(x=X, y=y, batch_size=1, verbose=1)
    print("loss :" + str(scores[0]))
    print("acc  :" + str(scores[1]))


def divideData(data_list):
    X = []
    y = []
    for features, label in data_list:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, 200, 200, 1)#for single channle
    X = X / 255.0

    y = keras.utils.to_categorical(y, 3)
    return X, y


def createImageForTesting():
    print("The images and labels getting for training and testing\n")
    unique = os.listdir("Data/test")
    test_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\test\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)#for single channle
            img_array = cv2.resize(img_array, (200, 200))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) #for single channle
            test_data.append([img_array, class_num])

    random.shuffle(test_data)
    print("number of test images: ", len(test_data))
    return test_data


test = createImageForTesting()
X, y = divideData(test)
predict_acc(model, X, y)

