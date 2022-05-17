import os
import random
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
import pygame as py
import Animation
from tensorflow.keras.models import load_model
import json
from processImage import model_recognizeSkin

bgcolor = "#F7F268"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

py.mixer.init()
py.mixer.music.load(r'assets/bgm/main_theme.mp3')

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

# with open('../skin_v4_1_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../skin_v4_1_model.h5")

# model = load_model("../skin_resnet_v3_model40.h5")
# model = load_model("../skin_alexnet_v1_model.h5")
model = load_model("../skin_v4_real_res_2_model.h5")


# with open('../resnet_model_1.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../resnet_model_1.h5")

# with open('../resnet_model_1.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model = load_model("../best_model.h5")

# with open('../skin_v3_simple_dataset_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../skin_v3_simple_dataset_model.h5")

# with open('../skin_v4_5_model.json', 'r') as f:
#     model_json = f.read()
# model = model_from_json(model_json)
# model.load_weights("../skin_v4_5_model.h5")


# def predict_result(processImage):
#
#     result_list = ["paper", "rock", "scissors"]
#
#     # processImage = recognizeSkin(processImage)
#     processImage = model_recognizeSkin(processImage)
#     processImage = cv.cvtColor(processImage, cv.COLOR_BGR2BGRA)
#     processImage = cv.resize(processImage, (200, 200), interpolation=cv.INTER_AREA)
#     processImage = image.img_to_array(processImage)
#     processImage = np.expand_disssms(processImage, axis=0)
#     result = model.predict(processImage)
#
#     result_index = np.argmax(result)
#     print(result_list[result_index])

def predict_result(img):
    result_list = ["paper", "rock", "scissors"]
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    processImage = recognizeSkin(img)
    plt.imshow(processImage)
    plt.show()
    # processImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    processImage = cv.resize(processImage, (200, 200), interpolation=cv.INTER_AREA)

    processImage = image.img_to_array(processImage)
    processImage = processImage / 255.0
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


# def predict_result(img):
#     result_list = ["paper", "rock", "scissors"]
#     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#     img_array = cv.resize(img, (200, 200))
#     processImage = recognizeSkin(img_array)
#     processImage = processImage / 255.0
#     plt.imshow(processImage)
#     plt.show()
#     picture = np.expand_dims(processImage, axis=0)
#     result = model.predict(picture)
#     print(result)
#     result_index = np.argmax(result)
#     result = result_list[result_index]
#     return result


def get_heightest_score():
    with open("assets/config.json", 'r') as f:
        data = json.load(f)
    score = data["score"]
    # f = open("assets/score.txt", "r")
    # score = f.read()
    # f.close()
    return score


def set_heightest_score(score):

    with open("assets/config.json", 'r') as f:
        data = json.load(f)
    data["score"] = score
    data = json.dumps(data)
    with open("assets/config.json", 'w') as w:
        w.write(data)

def mute(flag):
    with open("assets/config.json", 'r') as f:
        data = json.load(f)

    if flag == "True":
        data["mute"] = "True"
    else:
        data["mute"] = "False"

    data = json.dumps(data)
    with open("assets/config.json", 'w') as w:
        w.write(data)


def root_window_run():
    def bgm_btn(event):
        if mute_label['text'] == "Mute":
            py.mixer.music.stop()
            mute("True")
            mute_label.configure(image=img2) #stop play
            mute_label['text'] = "Unmute"
        else:
            mute("False")
            mute_label.configure(image=img1) #play
            mute_label['text'] = "Mute"
            py.mixer.music.play(-1, 10)

    def pve_run():
        root_window.destroy()
        pve_window()

    def imps_run():
        root_window.destroy()
        imps_window()

    def pvp_run():
        root_window.destroy()
        pvp_prepare_window()

    root_window = tk.Tk()
    root_window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING')
    root_window.geometry('1280x720')
    # root_window.geometry('1920x1080')
    root_window["background"] = "#F7F268"

    py.mixer.music.play(-1, 10)
    img1 = tk.PhotoImage(file="assets/mute.png")
    img2 = tk.PhotoImage(file="assets/unmute.png")

    # lines down below is about create an gif anime in main page
    numIdx = 9  # fram number of gif
    # fill 9 frames to "frames"
    frames = [tk.PhotoImage(file='assets/main_top.gif', format='gif -index %i' % (i)) for i in range(numIdx)]

    def update(idx):  # timer function for gif animate
        frame = frames[idx]
        idx += 1  # index of frame number：iterate 9 frames
        anime_label.configure(image=frame)  # show the current frame image
        root_window.after(200, update, idx % numIdx)  # continue after 0.2s

    # create anime label
    anime_label = tk.Label(root_window, bg="#F7F268", width=200, height=200)
    anime_label.place(relx=0.5, rely=0.2, anchor="center")
    root_window.after(0, update, 0)

    # create mute label
    mute_label = tk.Label(root_window, image=img1, text="Mute", font=("Showcard Gothic", 15), bg="#F7F268")
    mute_label.place(relx=0.05, rely=0.05, anchor="center")
    mute_label.bind("<Button-1>", bgm_btn)

    with open("assets/config.json") as f:
        data = json.load(f)
    if data["mute"] == "True":
        py.mixer.music.stop()
        mute_label.configure(image=img2)  # mute
        mute_label['text'] = "Unmute"
    else:
        mute_label.configure(image=img1)  # play
        mute_label['text'] = "Mute"
        py.mixer.music.play(-1, 10)


    # set memu background
    # background = ImageTk.PhotoImage(file="assets/background.jpg")
    # bglabel = tk.Label(root_window, image=background).pack()
    background = tk.Label(root_window, text="Rock Paper Scissor", fg='RoyalBlue', bg="#F7F268",
                          font=("Eras Bold ITC", 65), justify='center').place(relx=0.5, rely=0.45, anchor="center")

    # set score
    score = "Highest Score: " + str(get_heightest_score())
    board = tk.Message(text=score, width=300, font=("Copperplate Gothic Bold", 30, "italic"), bg="#F7F268").place(
        relx=0.8, rely=0)

    # three buttons
    rank_start_button = tk.Button(text="vs Computer", command=pve_run, relief="groove", font=("Eras Bold ITC", 30),
                                  fg="#4876FF", bg="#f0f0f0", width=10, )
    rank_start_button.place(relx=0.5, rely=0.62, anchor="center")
    rank_start_button.bind("<Motion>", movement)
    rank_start_button.bind("<Leave>", leave)

    pvp_start_button = tk.Button(text="vs Players", command=pvp_run, relief="groove", font=("Eras Bold ITC", 30),
                                 fg="#4876FF", bg="#f0f0f0", width=10)
    pvp_start_button.place(relx=0.5, rely=0.75, anchor="center")
    pvp_start_button.bind("<Motion>", movement)
    pvp_start_button.bind("<Leave>", leave)

    imps_start_button = tk.Button(text="IMPOSSIBLE", command=imps_run, relief="groove", font=("Eras Bold ITC", 30),
                                  fg="#4876FF", bg="#f0f0f0", width=10)
    imps_start_button.place(relx=0.5, rely=0.88, anchor="center")
    imps_start_button.bind("<Motion>", movement)
    imps_start_button.bind("<Leave>", leave)

    root_window.mainloop()


def imps_window():
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Rank')
    # window.geometry('1280x720')
    window.attributes('-fullscreen', True)
    window["background"] = "#F7F268"
    window.focus_force()

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=75, height=75,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.1, anchor="center")

    canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)  # camera canvas
    canvas.config(highlightthickness=0)
    canvas.place(relx=0.8, rely=0.5, anchor="center")

    right_hand_canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)  # camera canvas
    right_hand_canvas.config(highlightthickness=0)

    capture = cv.VideoCapture(0)

    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")

    rock_img = Image.open("assets/rock.png")
    rock_img_canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)
    rock_img_canvas.config(highlightthickness=0)
    rock_img_canvas.place(rely=0.5, anchor="w")
    # left_hand_in(rock_img_canvas, rock_img)
    counting = False  # set lock, avoid repeatly play

    def count_down_start(event):
        nonlocal counting
        if counting == False:
            rock_img_canvas.delete("all")
            img = cv_image(capture)
            result = predict_result(img)

            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)


            # nonlocal capture
            canvas.delete("all")
            if result == "rock":
                left_hand_res = "paper"
                left_hand = [rock_img_canvas, left_hand_res]
                right_hand = [canvas, result]
                right_hand_canvas.place(relx=0.85, rely=0.5, anchor="center")
                two_hand_in(rock_img_canvas, right_hand_canvas, left_hand_res, result)
                right_hand_canvas.place_forget()
                # two_hand_in(rock_img_canvas, canvas, "paper", result)

                # # left_gesture(rock_img_canvas, "paper")
            if result == "paper":
                left_hand_res = "scissors"
                left_hand = [rock_img_canvas, left_hand_res]
                right_hand = [canvas, result]
                right_hand_canvas.place(relx=0.85, rely=0.5, anchor="center")
                two_hand_in(rock_img_canvas, right_hand_canvas, left_hand_res, result)
                right_hand_canvas.place_forget()
                # two_hand_in(rock_img_canvas, canvas, "scissors", ressult)
                # # left_gesture(rock_img_canvas, "scissors")

            if result == "scissors":
                left_hand_res = "rock"
                left_hand = [rock_img_canvas, left_hand_res]
                right_hand = [canvas, result]
                right_hand_canvas.place(relx=0.85, rely=0.5, anchor="center")
                two_hand_in(rock_img_canvas, right_hand_canvas, left_hand_res, result)
                right_hand_canvas.place_forget()

                # two_hand_in(rock_img_canvas, canvas, "rock", result)

                # left_gesture(rock_img_canvas, "rock")
            tmp = [0]
            Animation.pk_result(window, tmp)
            counting = False  # unlock
        else:
            return

    capture_btn = tk.Button(text="Press S", relief="groove", font=("Eras Bold ITC", 30), bg="#f0f0f0", width=10)
    capture_btn.place(relx=0.5, rely=0.93, anchor="center")
    capture_btn.bind("<Button-1>", count_down_start)
    capture_btn.bind("<Motion>", movement)
    capture_btn.bind("<Leave>", leave)

    def main():
        while True:
            img = cv_image(capture)
            window.bind('<KeyPress-s>', count_down_start)
            picture = mt_player_tk_image(capture, 50, 50, 600, False)
            canvas.create_image(0, 0, anchor='nw', image=picture)
            window.update()
            window.after(100)

    main()


def pve_window():
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Rank')
    # window.geometry('1280x720')
    window.attributes('-fullscreen', True)
    window["background"] = "#F7F268"
    window.focus_force()

    # left_score = 0
    # right_score = 0
    # score = str(left_score) + " : " + str(right_score)
    score = 0
    score_board = tk.Label(window, text=str(score), background=bgcolor, fg="#707070", font=("Gabriola", 100))
    score_board.place(relx=0.5, rely=0.1, anchor="center")

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=75, height=75,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.1, anchor="center")

    canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)  # camera canvas
    canvas.config(highlightthickness=0)
    canvas.place(relx=0.8, rely=0.5, anchor="center")

    right_hand_canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)  # camera canvas
    right_hand_canvas.config(highlightthickness=0)


    capture = cv.VideoCapture(0)

    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")


    rock_img_canvas = tk.Canvas(window, bg=bgcolor, width=600, height=600)
    rock_img_canvas.config(highlightthickness=0)
    rock_img_canvas.place(rely=0.5, anchor="w")
    # left_hand_in(rock_img_canvas, rock_img)
    counting = False  # set lock, avoid repeatly play

    def count_down_start(event):
        nonlocal counting
        if counting == False:
            rock_img_canvas.delete("all")
            img = cv_image(capture)
            result = predict_result(img)

            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)


            # nonlocal capture
            gesture = ["rock", "paper", "scissors"]
            randon_index = random.randint(0, 2)
            canvas.delete("all")
            left_hand_res = gesture[randon_index]
            left_hand = [rock_img_canvas, left_hand_res, False]
            right_hand = [canvas, result, False]
            right_hand_canvas.place(relx=0.85, rely=0.5, anchor="center")
            two_hand_in(rock_img_canvas, right_hand_canvas, gesture[randon_index], result)
            right_hand_canvas.place_forget()
            # hands_in(left_hand, right_hand, players=2)
            res = who_win([left_hand_res, result], num=2)
            Animation.pk_result(window, res)
            print("winner: ", res)
            nonlocal score
            if res[0] == 0:
                score = 0
                score_board.config(text=score)
            elif res[0] == 1:
                score += 1
                height_score = get_heightest_score()
                score_board.config(text=score)
                if score > int(height_score):
                    set_heightest_score(str(score))
                    score_board.config(text=score, fg="#FF0000")
            counting = False  # unlock
            # two_hand_in(rock_img_canvas, canvas, gesture[randon_index], result)
        else:
            return

    capture_btn = tk.Button(text="Press S", relief="groove", font=("Eras Bold ITC", 30), bg="#f0f0f0", width=10)
    capture_btn.place(relx=0.5, rely=0.93, anchor="center")
    capture_btn.bind("<Button-1>", count_down_start)
    capture_btn.bind("<Motion>", movement)
    capture_btn.bind("<Leave>", leave)

    def main():
        while True:
            simg = cv_image(capture)
            window.bind('<KeyPress-s>', count_down_start)
            picture = mt_player_tk_image(capture, 50, 50, 600, False)
            canvas.create_image(0, 0, anchor='nw', image=picture)
            window.update()
            window.after(100)

    main()


def pvp_prepare_window():
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- Choose number of player')
    window.geometry('1280x720')
    window["background"] = "#F7F268"
    window.focus_set()

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=75, height=75,
              relief="raised", borderwidth=0).place(relx=0.05, rely=0.1, anchor="center")

    def two_person_run():
        window.destroy()
        pvp_run(2)

    def three_person_run():
        window.destroy()
        pvp_run(3)

    def four_person_run():
        window.destroy()
        pvp_run(4)

    one_p = tk.Button(text="2 player", command=two_person_run, font=("Eras Bold ITC", 30), fg="#4876FF", bg="#f0f0f0",
                      width=10,
                      relief="raised")
    one_p.place(relx=0.5, rely=0.2, anchor="center")
    one_p.bind("<Motion>", movement)
    one_p.bind("<Leave>", leave)

    two_p = tk.Button(text="3 player", command=three_person_run, font=("Eras Bold ITC", 30), fg="#4876FF", bg="#f0f0f0",
                      width=10,
                      relief="raised")
    two_p.place(relx=0.5, rely=0.5, anchor="center")
    two_p.bind("<Motion>", movement)
    two_p.bind("<Leave>", leave)

    three_p = tk.Button(text="4 player", command=four_person_run, font=("Eras Bold ITC", 30), fg="#4876FF",
                        bg="#f0f0f0", width=10,
                        relief="raised")
    three_p.place(relx=0.5, rely=0.8, anchor="center")
    three_p.bind("<Motion>", movement)
    three_p.bind("<Leave>", leave)
    window.mainloop()


def pvp_run(num):
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- PVP')
    window.attributes('-fullscreen', True)
    # window.geometry('1280x720')
    window["background"] = "#F7F268"
    window.focus_force()

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=65, height=65,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.05, anchor="center")

    canvas_width = 450
    canvas_height = 450

    left_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)
    left_top_hand_img_canvas.config(highlightthickness=0)
    # left_top_hand_img_canvas.place(rely=0.5, anchor="w")

    right_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)
    right_top_hand_img_canvas.config(highlightthickness=0)
    # right_top_hand_img_canvas.place(rely=0.5, anchor="w")

    right_bot_hand_img_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)
    right_bot_hand_img_canvas.config(highlightthickness=0)
    # right_bot_hand_img_canvas.place(rely=0.5, anchor="w")

    left_bot_hand_img_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)
    left_bot_hand_img_canvas.config(highlightthickness=0)
    # left_bot_hand_img_canvas.place(rely=0.5, anchor="w")





    left_top_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)  # camera canvas
    left_top_canvas.config(highlightthickness=0)

    right_top_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)  # camera canvas
    right_top_canvas.config(highlightthickness=0)

    right_bot_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)  # camera canvas
    right_bot_canvas.config(highlightthickness=0)

    left_bot_canvas = tk.Canvas(window, bg=bgcolor, width=canvas_width, height=canvas_height)  # camera canvas
    left_bot_canvas.config(highlightthickness=0)

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")



    if num == 2:
        left_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=600, width=600)
        left_top_hand_img_canvas.config(highlightthickness=0)
        # left_top_hand_img_canvas.place(relx=0.19, rely=0.5, anchor="center")

        right_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=600,  width=600)
        right_top_hand_img_canvas.config(highlightthickness=0)
        # right_top_hand_img_canvas.place(relx=0.85, rely=0.5, anchor="center")

        left_top_canvas = tk.Canvas(window, bg=bgcolor, width=700, height=700)
        left_top_canvas.config(highlightthickness=0)
        left_top_canvas.place(relx=0.22, rely=0.5, anchor="center")

        right_top_canvas = tk.Canvas(window, bg=bgcolor, width=700, height=700)  # camera canvas
        right_top_canvas.config(highlightthickness=0)
        right_top_canvas.place(relx=0.78, rely=0.5, anchor="center")

    if num == 3:
        # left_top_hand_img_canvas.place(relx=0.09, rely=0.5, anchor="center")
        # right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
        # right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")

        left_top_canvas.place(relx=0.18, rely=0.5, anchor="center")
        right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
        right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")

    if num == 4:
        # left_top_hand_img_canvas.place(relx=0.09, rely=0.3, anchor="center")
        # right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
        # right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")
        # left_bot_hand_img_canvas.place(relx=0.09, rely=0.75, anchor="center")

        left_top_canvas.place(relx=0.18, rely=0.3, anchor="center")
        left_bot_canvas.place(relx=0.18, rely=0.75, anchor="center")
        right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
        right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")


    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")
    counting = False
    winners = []
    losers = []
    lt_out = False
    rt_out = False
    lb_out = False
    rb_out = False

    def count_down_start(event):
        nonlocal counting
        if counting == False:

            # rock_img_canvas.delete("all")
            lt_img = mt_player_cv_image(capture, 50, 400)
            # cv.imshow("lt", lt_img)
            rt_img = mt_player_cv_image(capture, 50, 50)
            # cv.imshow("rt", rt_img)
            lb_img = mt_player_cv_image(capture, 280, 400)
            # cv.imshow("lb", lb_img)
            rb_img = mt_player_cv_image(capture, 280, 50)
            # cv.imshow("rb", rb_img)

            lt_res = predict_result(lt_img)
            rt_res = predict_result(rt_img)
            lb_res = predict_result(lb_img)
            rb_res = predict_result(rb_img)

            result = predict_result(img)

            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)


            # nonlocal captures

            # left_top_canvas.delete("all")
            # left_bot_canvas.delete("all")
            # right_top_canvas.delete("all")
            # right_bot_canvas.delete("all")
            if num >= 2:
                left_top_canvas.place_forget()
                right_top_canvas.place_forget()
            if num >= 3:
                right_bot_canvas.place_forget()
            if num == 4:
                left_bot_canvas.place_forget()


            nonlocal lt_out
            nonlocal rt_out
            nonlocal lb_out
            nonlocal rb_out

            for outer in losers: #set flag of users, if user out, set is true
                if outer == 0:
                    lt_out = True
                if outer == 1:
                    rt_out = True
                if outer == 2:
                    rb_out = True
                if outer == 3:
                    lb_out = True

            if lt_out: #if user out, set a paper to him, becasue hands in require four hand
                lt_res = "paper"
            if rt_out:
                rt_res = "paper"
            if lb_out:
                lb_res = "paper"
            if rb_out:
                rb_res = "paper"
            # if num == 2:
            #     left_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
            #     left_top_hand_img_canvas.config(highlightthickness=0)
            #     left_top_hand_img_canvas.place(relx=0.19, rely=0.5, anchor="center")
            #
            #     right_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
            #     right_top_hand_img_canvas.config(highlightthickness=0)
            #     right_top_hand_img_canvas.place(relx=0.85, rely=0.5, anchor="center")

            if num == 2:
                # left_top_canvas.place(relx=0.22, rely=0.5, anchor="center")
                # right_top_canvas.place(relx=0.78, rely=0.5, anchor="center")
                #
                left_top_hand_img_canvas.place(relx=0.09, rely=0.5, anchor="center")
                right_top_hand_img_canvas.place(relx=0.89, rely=0.5, anchor="center")

            if num == 3:
                left_top_hand_img_canvas.place(relx=0.09, rely=0.5, anchor="center")
                right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
                right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")

            if num == 4:
                left_top_hand_img_canvas.place(relx=0.09, rely=0.3, anchor="center")
                right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
                right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")
                left_bot_hand_img_canvas.place(relx=0.09, rely=0.75, anchor="center")

            lt_hand = [left_top_hand_img_canvas, lt_res, lt_out]
            rt_hand = [right_top_hand_img_canvas, rt_res, rt_out]
            lb_hand = [left_bot_hand_img_canvas, lb_res, lb_out]
            rb_hand = [right_bot_hand_img_canvas, rb_res, rb_out]

            print(lt_res, rt_res, rb_res, lb_res)

            hands_in(lt_hand, rt_hand, rb_hand, lb_hand, players=num)

            left_top_hand_img_canvas.place_forget() #hide hands canvas
            left_bot_hand_img_canvas.place_forget()
            right_top_hand_img_canvas.place_forget()
            right_bot_hand_img_canvas.place_forget()

            if num == 2:
                left_top_canvas.place(relx=0.22, rely=0.5, anchor="center")
                right_top_canvas.place(relx=0.78, rely=0.5, anchor="center")

            if num == 3:
                left_top_canvas.place(relx=0.18, rely=0.5, anchor="center")
                right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
                right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")
            if num == 4:
                left_top_canvas.place(relx=0.18, rely=0.3, anchor="center")
                left_bot_canvas.place(relx=0.18, rely=0.75, anchor="center")
                right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
                right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")

            nonlocal winners
            who_win_list = []
            if len(winners) > 0 and winners[0] != -1:
                for player in winners:
                    if player == 0:
                        who_win_list.append(lt_res)
                    elif player == 1:
                        who_win_list.append(rt_res)
                    elif player == 2:
                        who_win_list.append(rb_res)
                    elif player == 3:
                        who_win_list.append(lb_res)
            else:
                who_win_list = [lt_res, rt_res, rb_res, lb_res]

            print("winner is", winners)
            while len(who_win_list) > num:
                who_win_list.pop()
            new_winners_index = who_win(who_win_list, num=len(who_win_list))
            new_winners = []
            counting = False  # unlock
            if new_winners_index[0] == -1:
                return
            if len(winners) > 0 and winners[0] != -1:
                for new_winner in new_winners_index:
                    new_winners.append(winners[new_winner])
                winners = new_winners
            else:
                winners = new_winners_index
            print("winner", winners)


        else:
            return


    capture_btn = tk.Button(text="Press S", relief="groove", font=("Eras Bold ITC", 30), bg="#f0f0f0", width=10)
    capture_btn.place(relx=0.5, rely=0.93, anchor="center")
    capture_btn.bind("<Button-1>", count_down_start)
    capture_btn.bind("<Motion>", movement)
    capture_btn.bind("<Leave>", leave)

    capture = cv.VideoCapture(0)
    p1syb = ImageTk.PhotoImage(file="assets/p1.png")
    p2syb = ImageTk.PhotoImage(file="assets/p2.png")
    p3syb = ImageTk.PhotoImage(file="assets/p3.png")
    p4syb = ImageTk.PhotoImage(file="assets/p4.png")

    out = ImageTk.PhotoImage(file="assets/out.png")



    while True:

        resize = 700
        if num > 2:
            resize = 450
        img = cv_image(capture)
        left_top = mt_player_tk_image(capture, 50, 400, resize, False)
        right_top = mt_player_tk_image(capture, 50, 50, resize, False)
        left_bot = mt_player_tk_image(capture, 280, 400, resize, False)
        right_bot = mt_player_tk_image(capture, 280, 50, resize, False)

        map = {0: out, 1: out, 2: out, 3: out}
        if len(winners) > 0 and winners[0] != -1:
            loser_tmp = get_losser(winners)
            for i in loser_tmp:
                if i not in losers:
                    losers.append(i)
        if len(winners) == 1 and winners[0] != -1:
            left_top_canvas.place_forget()
            right_top_canvas.place_forget()
            left_bot_canvas.place_forget()
            right_bot_canvas.place_forget()
            Animation.muti_pk_result(window, winners)
            print("got winner", winners)
            losers =[]
            winners = []
            lt_out = False
            rt_out = False
            lb_out = False
            rb_out = False
            if num == 2:
                left_top_canvas.place(relx=0.22, rely=0.5, anchor="center")
                right_top_canvas.place(relx=0.78, rely=0.5, anchor="center")

            if num == 3:
                left_top_canvas.place(relx=0.18, rely=0.5, anchor="center")
                right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
                right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")
            if num == 4:
                left_top_canvas.place(relx=0.18, rely=0.3, anchor="center")
                left_bot_canvas.place(relx=0.18, rely=0.75, anchor="center")
                right_top_canvas.place(relx=0.82, rely=0.3, anchor="center")
                right_bot_canvas.place(relx=0.82, rely=0.75, anchor="center")




        print("losers", losers)
        for i in losers:
            if i == 0:
                left_top = mt_player_tk_image(capture, 50, 400, resize, True)

            if i == 1:
                right_top = mt_player_tk_image(capture, 50, 50, resize, True)

            if i == 2:
                right_bot = mt_player_tk_image(capture, 280, 50, resize, True)

            if i == 3:
                left_bot = mt_player_tk_image(capture, 280, 400, resize, True)


        left_top_canvas.create_image(0, 0, anchor='nw', image=left_top)
        right_top_canvas.create_image(0, 0, anchor='nw', image=right_top)
        left_bot_canvas.create_image(0, 0, anchor='nw', image=left_bot)
        right_bot_canvas.create_image(0, 0, anchor='nw', image=right_bot)

        if num == 2:
            left_top_canvas.create_image(635, 0, anchor='nw', image=p1syb)
            right_top_canvas.create_image(0, 0, anchor='nw', image=p2syb)
        else:
            left_top_canvas.create_image(385, 0, anchor='nw', image=p1syb)
            right_top_canvas.create_image(0, 0, anchor='nw', image=p2syb)
            left_bot_canvas.create_image(385, 0, anchor='nw', image=p3syb)
            right_bot_canvas.create_image(0, 0, anchor='nw', image=p4syb)

        x = 220
        y = 220
        size = 200
        if 0 in losers:
            left_top_canvas.create_text(x, y, anchor='center', text="OUT", font=("Gabriola", size),
                                        fill="#FF0000")
        if 1 in losers:
            right_top_canvas.create_text(x, y, anchor='center', text="OUT", font=("Gabriola", size),
                                         fill="#FF0000")
        if 2 in losers:
            right_bot_canvas.create_text(x, y, anchor='center', text="OUT", font=("Gabriola", size),
                                         fill="#FF0000")
        if 3 in losers:
            left_bot_canvas.create_text(x, y, anchor='center', text="OUT", font=("Gabriola", size),
                                        fill="#FF0000")
        window.bind('<KeyPress-s>', count_down_start)

        window.update()
        window.after(100)


def get_losser(winners):
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    for i in winners:
        if i == 0:
            flag1 = True
        if i == 1:
            flag2 = True
        if i == 2:
            flag3 = True
        if i == 3:
            flag4 = True

    losers = []
    if not flag1:
        losers.append(0)

    if not flag2:
        losers.append(1)

    if not flag3:
        losers.append(2)

    if not flag4:
        losers.append(3)

    return losers


def cv_image(capture):
    ref, frame = capture.read()
    frame = frame[50:250, 50:250]
    frame = cv.flip(frame, 180)
    cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return cvimage


def tk_image(capture):
    cvimage = cv_image(capture)
    pilImage = Image.fromarray(cvimage)

    pilImage = pilImage.resize((450, 450), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage


def mt_player_cv_image(capture, x, y, grey=False):
    ref, frame = capture.read()
    frame = frame[x:x + 200, y:y + 200]
    frame = cv.flip(frame, 180)
    if grey:
        cvimage = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    else:
        cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return cvimage


def mt_player_tk_image(capture, x, y, resize, grey):

    cvimage = mt_player_cv_image(capture, x, y, grey)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((resize, resize), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage


def count_down(canvas, image):
    minx = 50
    maxx = 250
    for i in range(minx, maxx + 1):
        image = image.resize((i, i * 2))
        tkimage = ImageTk.PhotoImage(image)
        canvas.create_image(180, 250, anchor='center', image=tkimage)
        canvas.update()
        # canvas.after(0.1)


def left_hand_in(canvas, image):
    global tkimage
    start = -50
    end = 180
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


def reconize_gesture(gesture):
    global img
    if gesture == "rock":
        img = Image.open("assets/rock.png")
    elif gesture == "paper":
        img = Image.open("assets/paper.png")
    elif gesture == "scissors":
        img = Image.open("assets/scissors.png")
    return img


def two_hand_in(left_canvas, right_canvas, left, right):
    global left_tkimage
    global right_tkimage
    step = 210

    left = reconize_gesture(left)
    right = reconize_gesture(right)
    right = right.rotate(180)
    right = right.transpose(Image.FLIP_TOP_BOTTOM)
    i = 0

    hi = 516
    len = 502
    lx = -10
    ly = 270

    rx = 600
    ry = 270

    while i < step:
        left_tkimage = ImageTk.PhotoImage(left)
        right_tkimage = ImageTk.PhotoImage(right)

        left_canvas.create_image(lx + i, ly, anchor='center', image=left_tkimage)
        right_canvas.create_image(rx - i, ry, anchor='center', image=right_tkimage)

        left_canvas.update()
        right_canvas.update()
        i += 2
    right_canvas.after(1000)
    right_canvas.delete("all")
    left_canvas.delete("all")
    # canvas.img = tkimage


def hands_in(*gestures, players):
    global lt_canvas, rt_canvas, rb_canvas, lb_canvas
    global lt_hand, rt_hand, rb_hand, lb_hand

    global lt_hand_image
    global rt_hand_image
    global rb_hand_image
    global lb_hand_image

    lt_canvas, lt_hand, lt_out = get_data_from_list(gestures, 0)
    rt_canvas, rt_hand, rt_out = get_data_from_list(gestures, 1)
    if players >= 3:
        rb_canvas, rb_hand, rb_out = get_data_from_list(gestures, 2)
    if players == 4:
        lb_canvas, lb_hand, lb_out = get_data_from_list(gestures, 3)

    hi = 495
    len = 495
    lx = 130
    ly = 270

    rx = 480
    ry = 270
    if players >= 3:
        hi = 380
        len = 380

        lx = 40
        ly = 190

        rx = 450
        ry = 190
    out = Image.open("assets/empty.png")
    if not lt_out:
        lt_hand = reconize_gesture(lt_hand).resize((hi, len))
    else:
        lt_hand = out
    if not rt_out:
        rt_hand = reconize_gesture(rt_hand).resize((hi, len))
        rt_hand = rotate_hand(rt_hand)
    else:
        rt_hand = out
    if players >= 3:
        if not rb_out:
            rb_hand = reconize_gesture(rb_hand).resize((hi, len))
            rb_hand = rotate_hand(rb_hand)
        else:
            rb_hand = out
    if players == 4:
        if not lb_out:
            lb_hand = reconize_gesture(lb_hand).resize((hi, len))
        else:
            lb_hand = out

    step = 210
    i = 0

    while i < step:
        lt_hand_image = ImageTk.PhotoImage(lt_hand)
        lt_canvas.create_image(lx + i, ly, anchor='center', image=lt_hand_image)
        lt_canvas.update()

        rt_hand_image = ImageTk.PhotoImage(rt_hand)
        rt_canvas.create_image(rx - i, ry, anchor='center', image=rt_hand_image)
        rt_canvas.update()

        if players >= 3:
            rb_hand_image = ImageTk.PhotoImage(rb_hand)
            rb_canvas.create_image(rx - i, ry, anchor='center', image=rb_hand_image)
            rb_canvas.update()

        if players == 4:
            lb_hand_image = ImageTk.PhotoImage(lb_hand)
            lb_canvas.create_image(lx + i, ly, anchor='center', image=lb_hand_image)
            lb_canvas.update()
        i += 2
    rt_canvas.after(1000)
    try:
        rt_canvas.delete("all")
        lt_canvas.delete("all")
        rb_canvas.delete("all")
        lb_canvas.delete("all")
    except:
        pass


def get_data_from_list(list, num):
    i = 0
    for k in list:
        if i == num:
            return k[0], k[1], k[2]
        i += 1
        if i >= len(list):
            return


def who_win(player_hand, num):
    hand_list = {}
    index = 1
    count = 0
    num_count = 0

    winner_list = []
    for p in player_hand:  # if there are more than 2 hands and only 1 hand, dead heat
        if p not in hand_list.values():
            hand_list[index] = p
            count += 1

        if count > 2:
            winner_list.append(-1)
            return winner_list
        index += 1
        num_count += 1
        if num_count > num:
            break

    if count == 1:
        winner_list.append(-1)
        return winner_list

    player_list = ['lt', 'rt', 'rb', 'lb']  # only two hand, then check who win
    player_hand_table = {}
    for i in range(num):
        player_hand_table[player_list[i]] = player_hand[i]
    win_hand = player_hand_table['lt']
    lose_hand = ""

    for h in player_hand_table.values():
        if h != win_hand:
            lose_hand = h

    if compare_hand(win_hand, lose_hand) == 2:
        tmp = win_hand
        win_hand = lose_hand
        lose_hand = tmp

    index = 0
    for hand in player_hand_table.values():
        if hand == win_hand:
            winner_list.append(index)
        index += 1
    return winner_list


def compare_hand(hand_a, hand_b):
    hand_table = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

    if hand_a == hand_b:
        return -1

    if hand_table[hand_a] == hand_b:
        return 1
    else:
        return 2


# when mouse move to button, it turns bigger
def movement(event):
    event.widget['font'] = ("Eras Bold ITC", 33)


# when mouse move to button, it change smaller
def leave(event):
    event.widget['font'] = ("Eras Bold ITC", 30)


def rotate_hand(hand):
    right = hand.rotate(180)
    return right.transpose(Image.FLIP_TOP_BOTTOM)


root_window_run()
