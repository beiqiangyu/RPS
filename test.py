import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import processImage
import os
from skimage import feature
from skimage import io
from processImage import recognizeSkin

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img = cv2.imread("test/paper.jpg", cv2.IMREAD_COLOR_)
img = recognizeSkin(img)



