import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def getImageNamesAndClassLabels():
    class_list = os.listdir('Dataset')
    image_names = []
    labels = []
    for class_name in class_list:
        image_list = os.listdir(os.path.join('Dataset', class_name))

        for image_name in image_list:
            image_names.append(image_name)
            labels.append(class_name)

    return image_names, labels

def splitDatasetTrainAndTest(image_names, labels):
    print("\nThe images are divided into test, train and validation folder")
    train_images, test_images, train_labels, test_labels = train_test_split(image_names, labels, shuffle = True, test_size=0.2)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, shuffle = True, test_size=0.25, random_state=1)
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def prepareTrainAndTestFolder(path, prev_path, images, labels):
    for i in range(0, len(images)):
        img = cv2.imread(os.path.join(prev_path, labels[i], images[i]), 1)
        img = recognizeSkin(img)

        if not (os.path.isfile(os.path.join(path, labels[i], images[i]))):
            cv2.imwrite(os.path.join(path, labels[i], images[i]), img)

def model_recognizeSkin(img):
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    imageYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    newimage = cv2.bitwise_and(img, img, mask=skinRegionYCrCb)
    resizedImg = cv2.resize(newimage, (200, 200), interpolation=cv2.INTER_AREA)
    return resizedImg

def recognizeSkin(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("Skin Cr+OTSU", skin)

    return skin

    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # (y, cr, cb) = cv2.split(ycrcb)
    #
    # skin = np.zeros(cr.shape, dtype=np.uint8)
    # (x, y) = cr.shape
    # for i in range(0, x):
    #     for j in range(0, y):
    #         if (cr[i][j] > 133) and (cr[i][j]) < 173 and (cb[i][j] > 77) and (cb[i][j]) < 127:
    #             skin[i][j] = 255
    #         else:
    #             skin[i][j] = 0
    # return skin

def createImageForTrainingAndTesting():
    print("The images and labels getting for training and testing\n")
    unique = os.listdir("Dataset")
    train_data = []
    for label in unique:

        class_num = unique.index(label)
        path = 'Data\\train\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            train_data.append([img_array, class_num])

    validation_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\validation\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            validation_data.append([img_array, class_num])

    test_data = []
    for label in unique:
        class_num = unique.index(label)
        path = 'Data\\test\\' + label
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), 1)
            test_data.append([img_array, class_num])

    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    # print("number of train images: ", len(train_data))
    # print("number of validation images: ", len(validation_data))
    # print("number of test images: ", len(test_data))

    return train_data, test_data, validation_data

def divideDataAndLabel(data_list):
    X = []
    y = []
    for features, label in data_list:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 200, 200, 3)
    X = X / 255.0
    y = keras.utils.to_categorical(y, 3)
    return X, y

def createFolders():
    labels = os.listdir("Dataset")
    path="Data"
    if not (os.path.exists(path)):
        os.makedirs(path)

    under_data_folder = ['train', 'test', 'validation']
    for folder_name in under_data_folder:
        if not (os.path.exists(os.path.join(path, folder_name))):
            os.makedirs(os.path.join(path, folder_name))

        for label in labels:
            if not (os.path.exists(os.path.join(path, folder_name, label))):
                os.makedirs(os.path.join(path, folder_name, label))
                print("create", folder_name, label)

def preProcessDataSet(): #run when Data floder not made, it will create floder and fill image
    if not(os.path.exists('Data')):
        names, labels = getImageNamesAndClassLabels()
        X_train, y_train, X_test, y_test, X_val, y_val = splitDatasetTrainAndTest(names, labels)
        createFolders()
        prepareTrainAndTestFolder('Data\\train', 'Dataset', X_train, y_train)
        prepareTrainAndTestFolder('Data\\test', 'Dataset', X_test, y_test)
        prepareTrainAndTestFolder('Data\\validation', 'Dataset', X_val, y_val)
        print("done")

def getDataSet():

    train, test, validation = createImageForTrainingAndTesting()
    X_train, y_train = divideDataAndLabel(train)
    X_test, y_test = divideDataAndLabel(test)
    X_val, y_val = divideDataAndLabel(validation)

    # print(np.shape(train))
    # print(np.shape(test))
    # print(np.shape(validation))
    # print(X_train)
    return X_train, y_train, X_test, y_test, X_val, y_val
# preProcessDataSet()
# getDataSet()

