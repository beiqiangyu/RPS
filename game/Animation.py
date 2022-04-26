import time
import tkinter as tk

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
