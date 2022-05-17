import time
import tkinter as tk

from PIL import ImageTk, Image

bgcolor = "#F7F268"
def pk_result(window, result):
    result_canvas = tk.Canvas(window, bg=bgcolor, height=230, width=1280)
    result_canvas.config(highlightthickness=0)
    result_canvas.place(relx=0.5, rely=0.5, anchor="center")
    min = 100
    max = 200

    if result[0] == 0:
        while(min < max):
            result_canvas.delete("all")
            text = result_canvas.create_text(640, 115, anchor='center', text="YOU LOSS", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()

    elif result[0] == 1:
        while(min < max):
            result_canvas.delete("all")
            text = result_canvas.create_text(640, 115, anchor='center', text="YOU WIN", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()

    elif result[0] == -1:
        while(min < max):
            result_canvas.delete("all")
            text = result_canvas.create_text(640, 115, anchor='center', text="DEAD HEAT", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()

    time.sleep(2)
    result_canvas.destroy()

def muti_pk_result(window, result):
    result_canvas = tk.Canvas(window, bg=bgcolor, height=230, width=1280)
    result_canvas.config(highlightthickness=0)
    result_canvas.place(relx=0.5, rely=0.5, anchor="center")
    min = 100
    max = 200

    p1 = Image.open("assets/p1.png")
    p2 = Image.open("assets/p2.png")
    p3 = Image.open("assets/p4.png")
    p4 = Image.open("assets/p3.png")

    # p1syb = ImageTk.PhotoImage(p1syb)
    # p2syb = ImageTk.PhotoImage(p2syb)
    # p3syb = ImageTk.PhotoImage(p3syb)
    # p4syb = ImageTk.PhotoImage(p4syb)

    logox = 320
    logoy = 115

    textx = 830
    texty = 115

    if result[0] == 0:
        while(min < max):
            result_canvas.delete("all")

            p1syb = p1.resize((min, min))
            p1syb = ImageTk.PhotoImage(p1syb)

            logo = result_canvas.create_image(logox, logoy, anchor='center', image=p1syb)
            text = result_canvas.create_text(textx, texty, anchor='center', text="WIN", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()
    elif result[0] == 1:
        while(min < max):
            result_canvas.delete("all")

            p2syb = p2.resize((min, min))
            p2syb = ImageTk.PhotoImage(p2syb)

            logo = result_canvas.create_image(logox, logoy, anchor='center', image=p2syb)
            text = result_canvas.create_text(textx, texty, anchor='center', text="WIN", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()
    elif result[0] == 2:
        while(min < max):
            result_canvas.delete("all")

            p3syb = p3.resize((min, min))
            p3syb = ImageTk.PhotoImage(p3syb)

            logo = result_canvas.create_image(logox, logoy, anchor='center', image=p3syb)
            text = result_canvas.create_text(textx, texty, anchor='center', text="WIN", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()
    elif result[0] == 3:
        while(min < max):
            result_canvas.delete("all")

            p4syb = p4.resize((min, min))
            p4syb = ImageTk.PhotoImage(p4syb)

            logo = result_canvas.create_image(logox, logoy, anchor='center', image=p4syb)
            text = result_canvas.create_text(textx, texty, anchor='center', text="WIN", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()

    elif result[0] == -1:
        while(min < max):
            result_canvas.delete("all")
            text = result_canvas.create_text(640, 115, anchor='center', text="DEAD HEAT", font=("Gabriola", min), fill="#FF0000")
            min += 1
            result_canvas.update()
    time.sleep(2)
    result_canvas.destroy()
