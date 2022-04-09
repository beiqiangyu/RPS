import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.preprocessing import image

from processImage import recognizeSkin

with open('skin_v4_model.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights("skin_v4_model.h5")

def predict_result(img):
    result_list = ["paper", "rock", "scissors"]
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    processImage = recognizeSkin(img)

    # processImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    processImage = cv.resize(processImage, (200, 200), interpolation=cv.INTER_AREA)
    processImage = image.img_to_array(processImage)
    picture = np.expand_dims(processImage, axis=0)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # picture = model_recognizeSkin(img)
    # picture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
    # picture = cv.resize(picture, (200, 200), interpolation=cv.INTER_AREA)
    # picture = image.img_to_array(picture)
    # picture = np.expand_dims(picture, axis=0)

    result = model.predict(picture)
    print(result)
    result_index = np.argmax(result)
    result = result_list[result_index]
    return result

# img = cv2.imread("test/paper2.png")
img = cv.imread("test/rock.jpg")
img = recognizeSkin(img)
cv.imshow("a", img)
cv.waitKey()
print(predict_result(img))


