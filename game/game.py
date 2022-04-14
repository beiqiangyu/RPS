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

with open('../skin_v4_1_model.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights("../skin_v4_1_model.h5")


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
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    processImage = recognizeSkin(img)
    plt.imshow(processImage)
    plt.show()
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


def get_heightest_score():
    f = open("assets/score.txt", "r")
    score = f.read()
    f.close()
    return score


def root_window_run():
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
    # root_window["background"] = "#F7F268"

    # set memu background
    background = ImageTk.PhotoImage(file="assets/background.jpg")
    bglabel = tk.Label(root_window, image=background).pack()

    # set score
    score = "Highest Score: " + str(get_heightest_score())
    board = tk.Message(text=score, font=("Gabriola", 30, "italic"), bg="#F7F268").place(relx=0.8, rely=0)

    # set memu button
    pvest = ImageTk.PhotoImage(file="assets/pve.png")
    pvpst = ImageTk.PhotoImage(file="assets/pvp.png")
    impst = ImageTk.PhotoImage(file="assets/imp.png")

    rank_start_button = tk.Button(text="PVE", image=pvest, command=pve_run, bg="#F5F176", width=223, height=46,
                                  relief="raised", borderwidth=0).place(relx=0.5, rely=0.7, anchor="center")
    pvp_start_button = tk.Button(text="PVP", image=pvpst, command=pvp_run, bg="#F5F176", width=224, height=46,
                                 relief="ridge", borderwidth=0).place(relx=0.5, rely=0.8, anchor="center")
    imps_start_button = tk.Button(text="IMPOSSIBLE", image=impst, command=imps_run, bg="#F5F176", width=224, height=46,
                                  relief="ridge", borderwidth=0).place(relx=0.5, rely=0.9, anchor="center")

    root_window.mainloop()


def imps_window():
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Impossible')
    window.geometry('1280x720')
    window["background"] = "#F7F268"
    window.focus_set()

    canvas = tk.Canvas(window, bg=bgcolor, width=500, height=500)  # camera canvas
    canvas.config(highlightthickness=0)
    canvas.place(relx=0.85, rely=0.5, anchor="center")
    capture = cv.VideoCapture(0)

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=75, height=75,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.1, anchor="center")

    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")

    rock_img = Image.open("assets/rock.png")
    rock_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
    rock_img_canvas.config(highlightthickness=0)
    rock_img_canvas.place(rely=0.5, anchor="w")
    # left_hand_in(rock_img_canvas, rock_img)
    counting = False  # set lock, avoid repeatly play

    def count_down_start(event):
        nonlocal counting
        rock_img_canvas.delete("all")
        img = cv_image(capture)
        result = predict_result(img)
        if counting == False:
            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)
            counting = False  # unlock

            # nonlocal capture
            canvas.delete("all")
            if result == "rock":
                two_hand_in(rock_img_canvas, canvas, "paper", result)
                # left_gesture(rock_img_canvas, "paper")
            if result == "paper":
                two_hand_in(rock_img_canvas, canvas, "scissors", result)
                # left_gesture(rock_img_canvas, "scissors")
            if result == "scissors":
                two_hand_in(rock_img_canvas, canvas, "rock", result)
                # left_gesture(rock_img_canvas, "rock")
        else:
            return

    def main():
        while True:
            img = cv_image(capture)
            window.bind('<KeyPress-s>', count_down_start)
            picture = tk_image(capture)
            canvas.create_image(0, 0, anchor='nw', image=picture)
            window.update()
            window.after(100)

    main()


def pve_window():
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- RPS GAMING  Rank')
    window.geometry('1280x720')
    window["background"] = "#F7F268"
    window.focus_set()

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=75, height=75,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.1, anchor="center")

    canvas = tk.Canvas(window, bg=bgcolor, width=500, height=500)  # camera canvas
    canvas.config(highlightthickness=0)
    canvas.place(relx=0.85, rely=0.5, anchor="center")
    capture = cv.VideoCapture(0)

    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")

    rock_img = Image.open("assets/rock.png")
    rock_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
    rock_img_canvas.config(highlightthickness=0)
    rock_img_canvas.place(rely=0.5, anchor="w")
    # left_hand_in(rock_img_canvas, rock_img)
    counting = False  # set lock, avoid repeatly play

    def count_down_start(event):
        nonlocal counting
        rock_img_canvas.delete("all")
        img = cv_image(capture)
        result = predict_result(img)
        if counting == False:
            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)
            counting = False  # unlock

            # nonlocal capture
            gesture = ["rock", "paper", "scissors"]
            randon_index = random.randint(0, 2)
            canvas.delete("all")
            left_hand = [rock_img_canvas, gesture[randon_index]]
            right_hand = [canvas, result]
            hands_in(left_hand, right_hand, players=2)
            # two_hand_in(rock_img_canvas, canvas, gesture[randon_index], result)
        else:
            return

    def main():
        while True:
            img = cv_image(capture)
            window.bind('<KeyPress-s>', count_down_start)
            picture = tk_image(capture)
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

    two_person = ImageTk.PhotoImage(file="assets/2p.png")
    three_person = ImageTk.PhotoImage(file="assets/3p.png")
    four_person = ImageTk.PhotoImage(file="assets/4p.png")

    tk.Button(text="2p", image=two_person, command=two_person_run, bg="#F5F176", width=462, height=100,
              relief="raised", borderwidth=0).place(relx=0.5, rely=0.2, anchor="center")

    tk.Button(text="3p", image=three_person, command=three_person_run, bg="#F5F176", width=462, height=100,
              relief="raised", borderwidth=0).place(relx=0.5, rely=0.5, anchor="center")

    tk.Button(text="4p", image=four_person, command=four_person_run, bg="#F5F176", width=462, height=100,
              relief="raised", borderwidth=0).place(relx=0.5, rely=0.8, anchor="center")
    window.mainloop()


def pvp_run(num):
    window = tk.Tk()
    window.title('42028 Deep Learning and Convolutional Neural Network Assignment 3 -- PVP')
    window.geometry('1280x720')
    window["background"] = "#F7F268"
    window.focus_set()

    def back_run():
        window.destroy()
        root_window_run()

    bkbut = ImageTk.PhotoImage(file="assets/back.png")
    bkbut_start = tk.Button(text="back", image=bkbut, command=back_run, bg="#F5F176", width=65, height=65,
                            relief="raised", borderwidth=0).place(relx=0.05, rely=0.05, anchor="center")

    left_top_canvas = tk.Canvas(window, bg=bgcolor, width=300, height=300)  # camera canvas
    left_top_canvas.config(highlightthickness=0)

    right_top_canvas = tk.Canvas(window, bg=bgcolor, width=300, height=300)  # camera canvas
    right_top_canvas.config(highlightthickness=0)

    right_bot_canvas = tk.Canvas(window, bg=bgcolor, width=300, height=300)  # camera canvas
    right_bot_canvas.config(highlightthickness=0)

    left_bot_canvas = tk.Canvas(window, bg=bgcolor, width=300, height=300)  # camera canvas
    left_bot_canvas.config(highlightthickness=0)

    count_canvas = tk.Canvas(window, bg=bgcolor, height=500)  # count down canvas
    count_canvas.config(highlightthickness=0)
    count_canvas.place(relx=0.5, rely=0.5, anchor="center")

    left_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=300)
    left_top_hand_img_canvas.config(highlightthickness=0)
    # left_top_hand_img_canvas.place(rely=0.5, anchor="w")

    right_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=300)
    right_top_hand_img_canvas.config(highlightthickness=0)
    # right_top_hand_img_canvas.place(rely=0.5, anchor="w")

    right_bot_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=300)
    right_bot_hand_img_canvas.config(highlightthickness=0)
    # right_bot_hand_img_canvas.place(rely=0.5, anchor="w")

    left_bot_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=300)
    left_bot_hand_img_canvas.config(highlightthickness=0)
    # left_bot_hand_img_canvas.place(rely=0.5, anchor="w")

    if num == 2:
        # left_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
        # left_top_hand_img_canvas.config(highlightthickness=0)
        # left_top_hand_img_canvas.place(relx=0.19, rely=0.5, anchor="center")
        #
        # right_top_hand_img_canvas = tk.Canvas(window, bg=bgcolor, height=500)
        # right_top_hand_img_canvas.config(highlightthickness=0)
        # right_top_hand_img_canvas.place(relx=0.85, rely=0.5, anchor="center")

        left_top_canvas = tk.Canvas(window, bg=bgcolor, width=500, height=500)
        left_top_canvas.config(highlightthickness=0)
        left_top_canvas.place(relx=0.19, rely=0.5, anchor="center")

        right_top_canvas = tk.Canvas(window, bg=bgcolor, width=500, height=500)  # camera canvas
        right_top_canvas.config(highlightthickness=0)
        right_top_canvas.place(relx=0.85, rely=0.5, anchor="center")

    if num == 3:
        # left_top_hand_img_canvas.place(relx=0.09, rely=0.5, anchor="center")
        # right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
        # right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")

        left_top_canvas.place(relx=0.09, rely=0.5, anchor="center")
        right_top_canvas.place(relx=0.89, rely=0.3, anchor="center")
        right_bot_canvas.place(relx=0.89, rely=0.75, anchor="center")

    if num == 4:
        # left_top_hand_img_canvas.place(relx=0.09, rely=0.5, anchor="center")
        # left_bot_hand_img_canvas.place(relx=0.09, rely=0.75, anchor="center")
        # right_top_hand_img_canvas.place(relx=0.89, rely=0.3, anchor="center")
        # right_bot_hand_img_canvas.place(relx=0.89, rely=0.75, anchor="center")

        left_top_canvas.place(relx=0.09, rely=0.3, anchor="center")
        left_bot_canvas.place(relx=0.09, rely=0.75, anchor="center")
        right_top_canvas.place(relx=0.89, rely=0.3, anchor="center")
        right_bot_canvas.place(relx=0.89, rely=0.75, anchor="center")

    three = Image.open("assets/three.png")
    two = Image.open("assets/two.png")
    one = Image.open("assets/one.png")
    counting = False

    def count_down_start(event):
        nonlocal counting
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
        if counting == False:
            counting = True  # lock
            count_down(count_canvas, three)
            count_down(count_canvas, two)
            count_down(count_canvas, one)
            counting = False  # unlock

            # nonlocal captures

            left_top_canvas.delete("all")
            left_bot_canvas.delete("all")
            right_top_canvas.delete("all")
            right_bot_canvas.delete("all")


            lt_hand = [left_top_canvas, lt_res]
            rt_hand = [right_top_canvas, rt_res]
            lb_hand = [left_bot_canvas, lb_res]
            rb_hand = [right_bot_canvas, rb_res]

            print(lt_res, rt_res, rb_res, lb_res,)
            hands_in(lt_hand, rt_hand, rb_hand, lb_hand, players=num)
        else:
            return
        print(who_win(lt_res, rt_res, rb_res, lb_res, num=num))

    capture = cv.VideoCapture(0)
    while True:
        resize = 450
        if num > 2:
            resize = 300
        img = cv_image(capture)

        left_top = mt_player_tk_image(capture, 50, 400, resize)
        right_top = mt_player_tk_image(capture, 50, 50, resize)
        left_bot = mt_player_tk_image(capture, 280, 400, resize)
        right_bot = mt_player_tk_image(capture, 280, 50, resize)

        left_top_canvas.create_image(0, 0, anchor='nw', image=left_top)
        right_top_canvas.create_image(0, 0, anchor='nw', image=right_top)
        left_bot_canvas.create_image(0, 0, anchor='nw', image=left_bot)
        right_bot_canvas.create_image(0, 0, anchor='nw', image=right_bot)

        window.bind('<KeyPress-s>', count_down_start)

        window.update()
        window.after(100)


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

def mt_player_cv_image(capture, x, y):
    ref, frame = capture.read()
    frame = frame[x:x + 200, y:y + 200]
    frame = cv.flip(frame, 180)
    cvimage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return cvimage

def mt_player_tk_image(capture, x, y, resize):
    cvimage = mt_player_cv_image(capture, x, y)
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
    while i < step:
        left_tkimage = ImageTk.PhotoImage(left)
        right_tkimage = ImageTk.PhotoImage(right)

        left_canvas.create_image(-40 + i, 250, anchor='center', image=left_tkimage)
        right_canvas.create_image(470 - i, 250, anchor='center', image=right_tkimage)

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


    lt_canvas, lt_hand = get_data_from_list(gestures, 0)
    rt_canvas, rt_hand = get_data_from_list(gestures, 1)
    if players >= 3:
        rb_canvas, rb_hand = get_data_from_list(gestures, 2)
    if players == 4:
        lb_canvas, lb_hand = get_data_from_list(gestures, 3)

    # try:
    #     lt_canvas, lt_hand = get_data_from_list(gestures, 0)
    #     rt_canvas, rt_hand = get_data_from_list(gestures, 1)
    #     rb_canvas, rb_hand = get_data_from_list(gestures, 2)
    #     lb_canvas, lb_hand = get_data_from_list(gestures, 3)
    #
    # except:
    #     pass

    hi = 416
    len = 402
    lx = -40
    ly = 250

    rx = 470
    ry = 250
    if players >=3:
        hi = 280
        len = 280

        lx = -60
        ly = 140

        rx = 400
        ry = 140

    lt_hand = reconize_gesture(lt_hand).resize((hi, len))

    rt_hand = reconize_gesture(rt_hand).resize((hi, len))
    rt_hand = rotate_hand(rt_hand)

    if players >= 3:
        rb_hand = reconize_gesture(rb_hand).resize((hi, len))
        rb_hand = rotate_hand(rb_hand)
    if players == 4:
        lb_hand = reconize_gesture(lb_hand).resize((hi, len))

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
            return k[0], k[1]
        i += 1
        if i >= len(list):
            return

def who_win(*player_hand, num):
    hand_list = {}
    index = 1
    count = 0
    num_count = 0

    for p in player_hand:  # if there are more than 2 hands and only 1 hand, dead heat
        if p not in hand_list.values():
            hand_list[index] = p
            count += 1

        if count > 2:
            return 0
        index += 1
        num_count += 1
        if num_count > num:
            break

    if count == 1:
        return 0

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
    winner_list = []
    index = 0
    for hand in player_hand_table.values():
        if hand == win_hand:
            winner_list.append(index)
        index += 1
    return winner_list
def compare_hand(hand_a, hand_b):

    hand_table = {"rock":"scissors", "paper":"rock", "scissors":"paper"}

    if hand_a == hand_b:
        return 0

    if hand_table[hand_a] == hand_b:
        return 1
    else:
        return 2





def rotate_hand(hand):
    right = hand.rotate(180)
    return right.transpose(Image.FLIP_TOP_BOTTOM)


root_window_run()
