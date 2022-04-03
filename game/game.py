import os
import tkinter as tk
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from processImage import recognizeSkin
from concurrent.futures import ThreadPoolExecutor
import threading

from processImage import model_recognizeSkin
bgcolor = "#F7F268"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# with open('../model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../model.h5")

# with open('../raw_input_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights('../raw_input_model.h5')


# with open('../skin_v2_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../skin_v2_model.h5")

with open('../skin_v3_model.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights("../skin_v3_model.h5")

# with open('../skin_v3_simple_dataset_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../skin_v3_simple_dataset_model.h5")


# def predict_result(processImage):
#
#     result_list = ["paper", "rock", "scissors"]
#
#     # processImage = recognizeSkin(processImage)
#     processImage = model_recognizeSkin(processImage)
#     processImage = cv.cvtColor(processImage, cv.COLOR_BGR2BGRA)
#     processImage = cv.resize(processImage, (200, 200), interpolation=cv.INTER_AREA)
#     processImage = image.img_to_array(processImage)
#     processImage = np.expand_dims(processImage, axis=0)
#     result = model.predict(processImage)
#
#     result_index = np.argmax(result)
#     print(result_list[result_index])

def predict_result(img):
    result_list = ["paper", "rock", "scissors"]
    processImage = recognizeSkin(img)
    # processImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    processImage = cv.resize(processImage, (200, 200), interpolation=cv.INTER_AREA)
    plt.imshow(processImage)

    processImage = image.img_to_array(processImage)
    processImage = np.expand_dims(processImage, axis=0)
    result = model.predict(processImage)
    result_index = np.argmax(result)
    result = result_list[result_index]
    return result

def get_heightest_score():
    f = open("assets/score.txt", "r")
    score = f.read()
    f.close()
    return score

def root_window_run():

    def rank_run():
        root_window.destroy()
        rank_window()

    def imps_run():
        root_window.destroy()
        imps_window()

    root_window =tk.Tk()
    root_window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING')
    root_window.geometry('1280x720')
    # root_window.geometry('1920x1080')
    # root_window["background"] = "#F7F268"

    #set memu background
    background = ImageTk.PhotoImage(file="assets/background.jpg")
    bglabel = tk.Label(root_window, image=background).pack()

    #set score
    score = "Highest Score: " + str(get_heightest_score())
    board = tk.Message(text=score, font=("Gabriola", 30, "italic"), bg="#F7F268").place(relx=0.8, rely=0)


    #set memu button
    rankst = ImageTk.PhotoImage(file="assets/rs.png")
    impost = ImageTk.PhotoImage(file="assets/is.png")

    rank_start_button = tk.Button( text="RANK START", image=rankst, command=rank_run, bg="#F5F176", width=457, height=98, relief="raised", borderwidth=0).place(relx=0.2, rely=0.8, anchor="center")
    imps_start_button = tk.Button( text="IMPOSSIBLE START", image=impost, command=imps_run, bg="#F5F176", width=457, height=98, relief="ridge", borderwidth=0).place(relx=0.8, rely=0.8, anchor="center")
    root_window.mainloop()

def rank_window():
    rank_window = tk.Tk()
    rank_window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Rank')
    rank_window.geometry('1280x720')
    rank_window["background"] = "#F7F268"
    rank_window.focus_set()

    canvas = tk.Canvas(rank_window, bg='white', width=400, height=400) #camera canvas
    canvas.place(relx=0.85, rely=0.5, anchor="center")
    capture = cv.VideoCapture(0)


    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")

    count_canvas = tk.Canvas(rank_window, bg=bgcolor, height=500) # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")

    rock_img = Image.open("assets/rock.png")
    rock_img_canvas = tk.Canvas(rank_window, bg=bgcolor, height=500)
    rock_img_canvas.config(highlightthickness=0)
    rock_img_canvas.place(rely=0.5, anchor="w")
    # left_hand_in(rock_img_canvas, rock_img)
    counting = False # set lock, avoid repeatly play

    def count_down_start(event):
        nonlocal counting
        rock_img_canvas.delete("all")
        if counting == False:
            counting = True #lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)
            counting = False #unlock

            # nonlocal capture
            img = cv_image(capture)
            result = predict_result(img)

            if result == "rock":
                left_gesture(rock_img_canvas, "paper")
            if result == "paper":
                left_gesture(rock_img_canvas, "scissors")
            if result == "scissors":
                left_gesture(rock_img_canvas, "rock")
        else:
            return

    def main():
        while True:
            img = cv_image(capture)
            rank_window.bind('<KeyPress-s>', count_down_start)
            picture = tk_image(capture)
            canvas.create_image(0, 0, anchor='nw', image=picture)
            rank_window.update()
            rank_window.after(100)
    # main_thread = threading.Thread(target=main, args=())
    # main_thread.setDaemon(True)
    # main_thread.start()
    main()


def imps_window():
    imps_window = tk.Tk()
    imps_window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Impossible')
    imps_window.geometry('1280x720')

def cv_image(capture):
    ref, frame = capture.read()
    frame = frame[50:250,50:250]
    frame = cv.flip(frame, 180)
    cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    return cvimage

def tk_image(capture):
    cvimage = cv_image(capture)
    pilImage = Image.fromarray(cvimage)

    pilImage = pilImage.resize((450, 450), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage

def count_down(canvas, image):
    minx = 50
    maxx = 250


    for i in range(minx, maxx+1):
        image = image.resize((i, i*2))
        tkimage = ImageTk.PhotoImage(image)
        canvas.create_image(180, 250, anchor='center', image=tkimage)
        canvas.update()
        # canvas.after(0.1)



def left_hand_in(canvas, image):

    global tkimage
    start = -50
    end = 170
    tkimage = ImageTk.PhotoImage(image)

    for i in range(start, end):
        tkimage = ImageTk.PhotoImage(image)
        canvas.create_image(i, 250, anchor='center', image=tkimage)
        canvas.update()
    # canvas.img = tkimage

def left_gesture(canvas, gesture):
    img = ""
    if gesture == "rock":
        img = Image.open("assets/rock.png")
    elif gesture == "paper":
        img = Image.open("assets/paper.png")
    elif gesture == "scissors":
        img = Image.open("assets/scissors.png")

    left_hand_in(canvas, img)



root_window_run()





