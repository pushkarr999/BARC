import os.path
import tkinter as tk
from tkinter.filedialog import askopenfile, asksaveasfilename, askopenfilename
from tkinter import ttk
import numpy as np
import pandas as pd
import random
import Functions
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# initialize a Tkinter root object
root = tk.Tk()
p1 = tk.PhotoImage(file='icon.png')
root.iconphoto(False, p1)
# getting screen width and height of display
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
# setting tkinter window size
root.geometry("%dx%d" % (width, height))
root.title("Leak Detection")
ans = tk.StringVar()
plot_frame = tk.Frame(root)


def plot_all(data):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    tabControl = ttk.Notebook(plot_frame)
    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)
    tabControl.add(tab1, text='Tab 1')
    tabControl.add(tab2, text='Tab 2')
    tabControl.pack(side='bottom', fill="x")
    fig = Figure(figsize=(16, 4), dpi=100)
    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    # Plotting both the curves simultaneously
    for columns in data.iloc[:, 2:]:
        plot1.plot(data['time'], data[columns],
                   color="#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)]), label=columns)

    # Naming the x-axis, y-axis and the whole graph
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Counts")
    plot1.set_title("Detectors")

    # Adding legend, which helps us recognize the curve according to it's color
    plot1.legend()
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master=tab1)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(side='bottom', fill='x', anchor='w')

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   tab1)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack(side='bottom', fill='x', anchor='sw')
    plot_frame.pack(side='bottom', fill='x', anchor='w')

    ##For tab2
    fig2 = Figure(figsize=(16, 4), dpi=100)
    # adding the subplot
    plot2 = fig2.add_subplot(111)

    # plotting the graph
    # Plotting both the curves simultaneously
    plot2.plot(data['time'], data['d1'],
               color='blue', label=columns)

    # Naming the x-axis, y-axis and the whole graph
    plot2.set_xlabel("Time")
    plot2.set_ylabel("Counts")
    plot2.set_title("Injection Detector")

    # Adding legend, which helps us recognize the curve according to it's color
    plot2.legend()
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas2 = FigureCanvasTkAgg(fig2,
                                master=tab2)
    canvas2.draw()

    # placing the canvas on the Tkinter window
    canvas2.get_tk_widget().pack(side='bottom', fill='x', anchor='w')

    # creating the Matplotlib toolbar
    toolbar2 = NavigationToolbar2Tk(canvas2,
                                    tab2)
    toolbar2.update()

    # placing the toolbar on the Tkinter window
    canvas2.get_tk_widget().pack(side='bottom', fill='x', anchor='sw')
    # plot_frame.pack(side='bottom', fill='x', anchor='w')


def open_file():
    global data
    file = askopenfilename(parent=root, title="Choose a File",
                           filetypes=[('CSV files', '*.csv'), ('Text files', '*.txt')])
    filename_var.set('Opened : ' + file.split("/")[-1])
    splittedname = file.split('.', 1)
    print(splittedname[-1])
    if file:
        if splittedname[-1] == 'txt':
            # print("Text file")
            data = Functions.texttocsv(file)
        else:
            data = pd.read_csv(file)
        # print(data.head())
        # global ans
        # ans.set(Functions.data_leak(data))
        # print(ans)


CheckVar1 = tk.BooleanVar()


def show(lower_frame):
    global CheckVar1
    if CheckVar1.get():
        lower_frame.pack(side='bottom', anchor='nw', fill='x')
    else:
        lower_frame.pack_forget()


def save(peak_lbl):
    filepath = asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

    if not filepath:
        return

    with open(filepath, 'w') as output_file:
        text = peak_lbl.get('1.0', "end")
        output_file.write(text)


def detect():
    # print("Button Pressed")
    global CheckVar1
    lower_frame = tk.Frame(print_frame)
    c1 = tk.Checkbutton(print_frame, text="Show peak information", variable=CheckVar1,
                        command=lambda: show(lower_frame))
    sav_btn = tk.Button(print_frame, text="Save Peak Information", command=lambda: save(peak_lbl))
    scrollbar1 = tk.Scrollbar(lower_frame)
    scrollbar1.pack(side='right', fill='y')
    peak_lbl = tk.Text(lower_frame, padx=20, font=('arial', 12), yscrollcommand=scrollbar1.set)
    peak_lbl.pack(side='top', fill='x', anchor='nw')

    upper_frame = tk.Frame(print_frame, height=100, width=print_frame.winfo_width())
    scrollbar = tk.Scrollbar(upper_frame)
    scrollbar.pack(side='right', fill='y')
    ans_lbl = tk.Text(upper_frame, padx=20, font=('arial', 12), yscrollcommand=scrollbar.set)
    ans_lbl.pack(side='top', fill='x', anchor='nw')
    if threshold_var.get() or len(threshold_txt.get()) == 0:
        answer, inj_time, exit_time, threso, avgrad, peak_info = Functions.data_leak(data, 0, var.get(), var2.get())
    else:
        answer, inj_time, exit_time, threso, avgrad, peak_info = Functions.data_leak(data, int(threshold_txt.get()),
                                                                                     var.get(), var2.get())
    ans_lbl.insert('end', answer)
    ans_lbl.configure(state='disabled')
    scrollbar.config(command=ans_lbl.yview)
    upper_frame.pack(side='top', anchor='nw')
    sav_btn.pack(side='top', anchor='nw',padx=10)
    c1.pack(side='top', anchor='nw')

    peak_lbl.insert('end', peak_info)
    peak_lbl.configure(state='disabled')
    scrollbar1.config(command=peak_lbl.yview)

    upper_frame.pack_propagate(0)
    # print(peak_info)

    avgrad = np.around(avgrad, 3)
    exit_time = np.around(exit_time, 3)
    inj_time = np.around(inj_time, 3)
    threso = np.around(threso, 3)
    bgrad.set(bgrad.get() + str(avgrad))
    inj.set(inj.get() + str(inj_time))
    exittime.set(exittime.get() + str(exit_time))
    thres.set(thres.get() + str(threso))
    detectors.set(detectors.get() + str(len(data.columns)-1))


def clear():
    global CheckVar1
    bgrad.set("Background Radiation : ")
    thres.set("Applied Threshold : ")
    inj.set("Injection time : ")
    exittime.set("Exit Time : ")
    detectors.set("Number of detectors : ")
    CheckVar1.set(False)
    ans.set("")
    for widget in plot_frame.winfo_children():
        widget.destroy()
    for widget in print_frame.winfo_children():
        widget.destroy()


# browse button
open_frame = tk.Frame(root)
browse_text = tk.StringVar()
filename_var = tk.StringVar()
browse_btn = tk.Button(open_frame, textvariable=browse_text, command=lambda: open_file(), font=("Arial", 15),
                       relief='raised',
                       borderwidth=2)
browse_text.set("Open File")
file_lbl = tk.Label(open_frame, textvariable=filename_var, font=('Arial', 13), padx=10)
browse_btn.pack(side='left', anchor='nw')
file_lbl.pack(side='left')

plot_btn = tk.Button(root, text="Plot", command=lambda: plot_all(data))
frame = tk.Frame(root, height=400)
# Smoothing Frame
smoothing_frame = tk.LabelFrame(frame, text="Smoothing", font=("Arial", 15))
var2 = tk.IntVar()
check_var2 = tk.IntVar(value=1)
movavg = tk.Radiobutton(smoothing_frame, text="Moving Averages", padx=20, pady=20, variable=var2, value=1,
                        state="disabled")
savgol = tk.Radiobutton(smoothing_frame, text="Savitzky Golay Filter", padx=20, pady=20, variable=var2, value=2,
                        state="disabled")
convolve = tk.Radiobutton(smoothing_frame, text="Convolve", padx=20, pady=20, variable=var2, value=3, state="disabled")
lowess = tk.Radiobutton(smoothing_frame, text="Lowess", padx=20, pady=20, variable=var2, value=4, state="disabled")
exp = tk.Radiobutton(smoothing_frame, text="Exponentially Weighted Average", padx=20, pady=20, variable=var2, value=5,
                     state="disabled")


def smooth(choice):
    if choice == 1:
        var2.set(0)
        savgol.config(state="disabled")
        movavg.config(state="disabled")
        convolve.config(state="disabled")
        lowess.config(state="disabled")
        exp.config(state="disabled")
    else:
        savgol.config(state="normal")
        movavg.config(state="normal")
        convolve.config(state="normal")
        lowess.config(state="normal")
        exp.config(state="normal")


smoothing = tk.Checkbutton(smoothing_frame, text="Use Recommended", variable=check_var2, padx=20, pady=10,
                           command=lambda: smooth(check_var2.get()))

movavg.grid(column=0, row=0, sticky="nw")
savgol.grid(column=0, row=1, sticky="nw")
convolve.grid(column=0, row=2, sticky="nw")
lowess.grid(column=0, row=3, sticky="nw")
exp.grid(column=0, row=4, sticky="nw")
smoothing.grid(column=0, row=5, sticky="nw")

# Peak Detection
var = tk.IntVar()
check_var = tk.IntVar(value=1)
peak_frame = tk.LabelFrame(frame, text="Peak Detection Technique", font=("Arial", 15))

find_peak = tk.Radiobutton(peak_frame, text="Find Peaks Python", padx=20, pady=20, variable=var, value=1,
                           state="disabled")
cwt = tk.Radiobutton(peak_frame, text="Continous wavelet transform", padx=20, pady=20, variable=var, value=2,
                     state="disabled")
maxima = tk.Radiobutton(peak_frame, text="Local Maxima", padx=20, pady=20, variable=var, value=3, state="disabled")
tony = tk.Radiobutton(peak_frame, text="Peak to Average Ratio", pady=20, padx=20, variable=var, value=4,
                      state="disabled")


def detection(choice):
    if choice == 1:
        var.set(0)
        find_peak.config(state="disabled")
        cwt.config(state="disabled")
        maxima.config(state="disabled")
        tony.config(state="disabled")
    else:
        find_peak.config(state="normal")
        cwt.config(state="normal")
        maxima.config(state="normal")
        tony.config(state="normal")


detecting = tk.Checkbutton(peak_frame, text="Use Recommended", variable=check_var, padx=20, pady=20,
                           command=lambda: detection(check_var.get()))

find_peak.grid(column=0, row=0, sticky="nw")
cwt.grid(column=0, row=1, sticky="nw")
maxima.grid(column=0, row=2, sticky="nw")
tony.grid(column=0, row=3, sticky="nw")
detecting.grid(column=0, row=5, sticky="nw")

# Answer frame

ans_frame = tk.LabelFrame(frame, text="Answer", font=("Arial", 15), bd=5)
info_frame = tk.Frame(ans_frame)

print_frame = tk.Frame(ans_frame, highlightthickness=2, highlightbackground='grey')

threshold_frame = tk.Frame(info_frame)
threshold_lbl = tk.Label(threshold_frame, text="Enter Threshold", justify='left', anchor='w', padx=20, pady=15)


def threshold(txt):
    if threshold_var.get():
        threshold_txt.config(state='disable')
    else:
        threshold_txt.config(state='normal')


def open_win(data):
    new = tk.Toplevel(root)
    #new.geometry("750x250")
    new.title("Describe")
    # Create a Label in New window
    tk.Label(new, text=data.describe(), font='Arial 12').pack(pady=30)


threshold_txt = tk.Entry(threshold_frame, bg='light cyan')
threshold_var = tk.BooleanVar()
threshold_chk = tk.Checkbutton(threshold_frame, text="Select Automatically", variable=threshold_var,
                               command=lambda: threshold(threshold_txt.get()))

bgrad = tk.StringVar()
bgrad.set("Background Radiation : ")
thres = tk.StringVar()
thres.set("Applied Threshold : ")
inj = tk.StringVar()
inj.set("Injection time : ")
exittime = tk.StringVar()
exittime.set("Exit Time : ")
detectors = tk.StringVar()
detectors.set("Number of detectors : ")

bgrad_lbl = tk.Label(info_frame, textvariable=bgrad, justify='left', anchor='w', padx=20, pady=12)
thres_lbl = tk.Label(info_frame, textvariable=thres, justify='left', anchor='w', padx=20, pady=12)
inj_lbl = tk.Label(info_frame, textvariable=inj, justify='left', anchor='w', padx=20, pady=12)
exittime_lbl = tk.Label(info_frame, textvariable=exittime, justify='left', anchor='w', padx=20, pady=12)
detectors_lbl = tk.Label(info_frame, textvariable=detectors, justify='left', anchor='w', padx=20, pady=12)

btn_frame = tk.Frame(info_frame)
detect_btn = tk.Button(btn_frame, text="Detect", command=lambda: detect())
describe_btn = tk.Button(btn_frame, text="Describe", command=lambda: open_win(data))
describe_btn.pack(side='left', anchor='nw', padx=15)
detect_btn.pack(side='left', padx=15)
clear_btn = tk.Button(btn_frame, text="Clear", command=lambda: clear())
clear_btn.pack(side='right', padx=15)

threshold_chk.grid(column=1, row=1, sticky='w')
threshold_lbl.grid(column=0, row=0, sticky='w')
threshold_txt.grid(column=1, row=0, sticky='w')
threshold_frame.grid(column=0, row=0, sticky='w')
bgrad_lbl.grid(column=0, row=2, sticky='w')
thres_lbl.grid(column=0, row=3, sticky='w')
inj_lbl.grid(column=0, row=4, sticky='w')
exittime_lbl.grid(column=0, row=5, sticky='w')
detectors_lbl.grid(column=0, row=6, sticky='w')
btn_frame.grid(column=0, row=7, sticky='nsew', padx=15, pady=10)

info_frame.pack(side='left', anchor='nw')
print_frame.pack(side='right', anchor='ne', padx=(70, 20), pady=(10, 20), fill='both', expand='true')

# Grid Placing
open_frame.pack(side='top', anchor='nw', padx=20, pady=20)
plot_btn.pack(side='bottom', fill='x')
smoothing_frame.pack(side='left', anchor='nw', padx=20, expand='false', fill='y')
peak_frame.pack(side='left', anchor='nw', padx=20, expand='false', fill='y')
ans_frame.pack(side='left', anchor='nw', expand='true', fill='both', padx=20)
frame.pack(side='top', anchor='nw', fill='both')
frame.pack_propagate(0)
root.mainloop()
