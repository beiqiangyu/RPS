import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import processImage
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('model.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model =model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

img = plt.imread("test/scissors.jpg")
# picture = processImage.recognizeSkin(img)
# plt.imshow(picture)
# plt.show()
picture = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
picture = cv.resize(picture, (200, 200), interpolation=cv.INTER_AREA)
picture = image.img_to_array(picture)
picture = np.expand_dims(picture, axis=0)
result = loaded_model.predict(picture)
print(result)



# for i in range (0, 9):
#     name = "game/test " + str(i) + ".jpg"
#     img = plt.imread(name)
#     picture = processImage.recognizeSkin(img)
#     picture = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
#     picture = cv.resize(picture, (200, 200), interpolation=cv.INTER_AREA)
#     picture = image.img_to_array(picture)
#     picture = np.expand_dims(picture, axis=0)
#     result = model.predict(picture)
#     print(result)
