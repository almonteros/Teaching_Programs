# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:07:09 2021

@author: almon
"""

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from math import cos, sin
from sympy import Symbol
from sympy.utilities.lambdify import lambdify


pi = np.pi


def time(t_0, i=1):
    while True:
        yield t_0
        t_0 += i


class MainWindow(tk.Frame):
    def __init__(self, root):
        tk.Frame.__init__(self, root)
        self.root = root
        self.sim_window = tk.Frame(self.root)
        self.sim_window.pack(side=tk.TOP)

        # buttons
        button_container = tk.Frame(self.root)
        button_container.pack(side=tk.TOP, fill="both")

        start_button = tk.Button(
            button_container, text="Start", command=lambda: self._start())
        start_button.pack(side=tk.LEFT, expand=True)

        stop_button = tk.Button(
            button_container, text="Stop", command=lambda: self._stop())
        stop_button.pack(side=tk.RIGHT)

        # Inputs
        input_container = tk.Frame(self.root)
        input_container.pack(side=tk.TOP)

        # x(t)
        label_input = tk.StringVar()
        label_input.set("Input x(t) here, use t for time variable")
        label_input_box = tk.Label(
            input_container, textvariable=label_input, height=1)
        label_input_box.pack(side=tk.TOP)

        self.x_function = tk.StringVar()
        self.x_function.set("")
        x_input = tk.Entry(input_container, textvariable=self.x_function)
        x_input.pack(side=tk.TOP)

        # y(t)
        label_input = tk.StringVar()
        label_input.set("Input y(t) here, use t for time variable")
        label_input_boy = tk.Label(
            input_container, textvariable=label_input, height=1)
        label_input_boy.pack(side=tk.TOP)

        self.y_function = tk.StringVar()
        self.y_function.set("")
        y_input = tk.Entry(input_container, textvariable=self.y_function)
        y_input.pack(side=tk.TOP)

        lim_container = tk.Frame(self.root)
        lim_container.pack(side=tk.TOP, fill="both")

        # x limits
        x_lim_container = tk.Frame(lim_container)
        x_lim_container.pack(side=tk.LEFT, fill="both")

        label_xlim = tk.StringVar()
        label_xlim.set("x limits (min, max)")
        label_xlim_box = tk.Label(
            x_lim_container, textvariable=label_xlim, height=1)
        label_xlim_box.pack(side=tk.TOP)

        self.x_min_str = tk.StringVar()
        x_lim_input = tk.Entry(x_lim_container, textvariable=self.x_min_str)
        x_lim_input.pack(side=tk.TOP)

        self.x_max_str = tk.StringVar()
        x_lim_input = tk.Entry(x_lim_container, textvariable=self.x_max_str)
        x_lim_input.pack(side=tk.TOP)

        # y limits
        y_lim_container = tk.Frame(lim_container)
        y_lim_container.pack(side=tk.LEFT, fill="both")

        label_ylim = tk.StringVar()
        label_ylim.set("y limits (min max)")
        label_ylim_box = tk.Label(
            y_lim_container, textvariable=label_ylim, height=1)
        label_ylim_box.pack(side=tk.TOP)

        self.y_min_str = tk.StringVar()
        y_lim_input = tk.Entry(y_lim_container, textvariable=self.y_min_str)
        y_lim_input.pack(side=tk.TOP)

        self.y_max_str = tk.StringVar()
        y_lim_input = tk.Entry(y_lim_container, textvariable=self.y_max_str)
        y_lim_input.pack(side=tk.TOP)

        limit_button = tk.Button(lim_container, text="Set limits",
                                 command=lambda: self._set_limits())
        limit_button.pack(side=tk.LEFT, expand=True)

        self.pack()
        self.fig = plt.Figure()
        self.fig.set_figheight(8)
        self.fig.set_figwidth(8)
        self.x_min, self.x_max, self.y_min, self.y_max = -5, 5, -5, 5

    def _start(self):
        # function part
        self.x = lambda t: 0
        self.y = lambda t: 0

        # self.ax.scatter(0,0)
        x_function = self.x_function.get()

        y_function = self.y_function.get()

        t_var = Symbol('t')
        self.x = lambdify(t_var, x_function)
        self.y = lambdify(t_var, y_function)

        def update(t):
            x_val = self.x(t)
            y_val = self.y(t)
            self.xdata.append(x_val)
            self.ydata.append(y_val)
            self.line.set_data(self.xdata, self.ydata)
            return self.line,

        def init():
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)
            return self.line,
        self.ax = self.fig.add_subplot()  # plt.subplots(1,figsize=(8,8))
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlabel("x(t)")
        self.ax.set_ylabel("y(t)")
        self.ax.grid(True)
        self.xdata, self.ydata = [], []
        self.line, = self.ax.plot(0, 0, 'ro')
        ts = time(0, 0.1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(column=0, row=4)
        self.a = animation.FuncAnimation(self.fig, update, frames=ts)

    def _stop(self):
        self.line, = self.ax.plot(0, 0, 'ro')
        self.a.event_source.stop()
        self.xdata, self.ydata = [], []

    def _set_limits(self):
        lim_list = [self.x_min_str.get(), self.x_max_str.get(),
                    self.y_min_str.get(), self.y_max_str.get()]
        any_empty = [x == '' for x in lim_list]
        if any(any_empty):
            print("There is an empty limit")
        else:
            self.x_min, self.x_max, self.y_min, self.y_max = [
                float(x) for x in lim_list]
            self.ax.set_xlim(self.x_min, self.x_max)
            self.ax.set_ylim(self.y_min, self.y_max)


root = tk.Tk()
if __name__ == '__main__':
    app = MainWindow(root)
    app.mainloop()
    plt.close('all')
