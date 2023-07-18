import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFilter
import pickle
import numpy as np
from keras.models import load_model
from tkinter import StringVar
import pyperclip





# Create a dictionary to map predicted digits to symbols
symbol_dict = {
    '0': 'α',
    '1': 'β',
    '2': 'γ',
    '3': 'δ',
    '4': 'λ',
    '5': 'μ',
    '6': 'Ω',
    '7': 'π',
    '8': 'φ',
    '9': 'θ'
}

# Global variables
last_x, last_y = None, None
image_size = (25, 25)
drawing_points = []

def get_x_and_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw_smth(event):
    global last_x, last_y
    canvas_pred.create_line((last_x, last_y, event.x, event.y),
                       fill='black', width=4)
    last_x, last_y = event.x, event.y

    # Add the current point to the drawing points list
    drawing_points.append((event.x, event.y))

def clear_canvas():
    canvas_pred.delete("all")
    drawing_points.clear()
    digit_label.config(text="")

def clear_canvas_tab2():
    canvas_draw.delete("all")
    drawing_points.clear()

def draw_smth_tab2(event):
    global last_x, last_y
    canvas_draw.create_line((last_x, last_y, event.x, event.y),
                       fill='black', width=4)
    last_x, last_y = event.x, event.y

    # Add the current point to the drawing points list
    drawing_points.append((event.x, event.y))

def save_image():
    # Ask user to select a file path to save the image
    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        # Create a blank image with white background
        image = Image.new("RGB", (300, 300), "white")
        draw = ImageDraw.Draw(image)

        # Draw the lines from the drawing points onto the image
        for i in range(1, len(drawing_points)):
            x1, y1 = drawing_points[i-1]
            x2, y2 = drawing_points[i]
            draw.line((x1, y1, x2, y2), fill="black", width=6)

        # Resize and crop the image to the desired size
        image = image.resize(image_size, Image.ANTIALIAS)
        image = image.crop((0, 0, image_size[0], image_size[1]))
        image = image.filter(ImageFilter.SHARPEN)

        # Save the image to the selected file path
        image.save(file_path)
        print("Image saved successfully!")

def update_predicted_digit(digit):
    digit_label.config(text=symbol_dict[str(digit)])

def predict():
    image = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(image)

    # Draw the lines from the drawing points onto the image
    for i in range(1, len(drawing_points)):
        x1, y1 = drawing_points[i-1]
        x2, y2 = drawing_points[i]
        draw.line((x1, y1, x2, y2), fill="black", width=6)

    # Resize and crop the image to the desired size
    image = image.resize(image_size, Image.ANTIALIAS)
    image = image.crop((0, 0, image_size[0], image_size[1]))
    image = image.filter(ImageFilter.SHARPEN)

    model_name =  selected_model.get()

    if model_name == 'CNN model':
        model = 'my_model.h5'
    elif model_name == 'SVM model':
        model = 'MNIST_SVM.pickle'
    elif model_name == 'KNN model':
        model = 'MNIST_KNN.pickle'


    if model != "my_model.h5" :
        image_flat = np.array(image).flatten()
        with open( model, 'rb') as f:
            clf = pickle.load(f)
        digit_pred = clf.predict([image_flat])
        print("Predicted Digit:", symbol_dict[str(digit_pred[0])])
        update_predicted_digit(digit_pred[0])
   
    else :
        new_model =load_model(model)
        image = image.convert('L')
        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        digit_pred = new_model.predict(img_array)
        digit_pred = np.argmax(digit_pred, axis=1)
        print("Predicted Digit:", symbol_dict[str(digit_pred[0])])
        update_predicted_digit(digit_pred[0])

def copy_to_clipboard():
    digit = digit_label.cget("text")
    pyperclip.copy(digit)


               
# Create the main window
root = tk.Tk()
root.title("Nhận diện kí tự khó")
root.resizable(False, False)

selected_model = StringVar(root)
selected_model.set("CNN model")

# Create a tab control
tab_control = ttk.Notebook(root)

# Create a frame for the drawing tab
drawing_tab = ttk.Frame(tab_control)
pred_tab = ttk.Frame(tab_control)

# Add the drawing tab to the tab control
tab_control.add(pred_tab, text="Predict")
tab_control.add(drawing_tab, text="Drawing")

# Pack the tab control
tab_control.pack(expand=1, fill="both")

"""
PREDICT TAB
"""

# Create a frame to hold the canvas
canvas_frame = ttk.Frame(pred_tab)
canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Create a canvas for draw
canvas_pred = Canvas(canvas_frame, bg="white", width=300, height=300)
canvas_pred.pack(side="left", expand=True)

# Create a frame for the predicted digit
result_frame = ttk.Frame(canvas_frame)
result_frame.pack(side="right", padx=10)

# Create a label to display the predicted digit
result_label = ttk.Label(result_frame, text="Predicted Digit:")
result_label.pack(pady=10)

# Create a label to show the predicted digit
digit_label = ttk.Label(result_frame, text="", font=("Arial", 40))
digit_label.pack()

copy_button = ttk.Button(result_frame, text="Copy", command=copy_to_clipboard)
copy_button.pack(pady=10)



# Bind mouse events to canvas
canvas_pred.bind("<Button-1>", get_x_and_y)
canvas_pred.bind("<B1-Motion>", draw_smth)

# Create a frame for buttons
bottom_frame = ttk.Frame(pred_tab)
bottom_frame.pack(pady=10)

model_combobox = ttk.Combobox(bottom_frame, textvariable=selected_model)
model_combobox["values"] = ("CNN model","SVM model", "KNN model",)
model_combobox.pack(side="left", padx=10)

save_button = ttk.Button(bottom_frame, text="Save Image", command=save_image)
save_button.pack(side="left")

# Create a button to predict the image
pred_button = ttk.Button(bottom_frame, text="Predict", command=predict)
pred_button.pack(side="left", padx=10)

# Create a button to clear the canvas
clear_button = ttk.Button(
    bottom_frame, text="Clear All", command=clear_canvas)
clear_button.pack(side="left")

"""
DRAWING TAB
"""
canvas_draw = Canvas(drawing_tab, bg="white")
canvas_draw.place(relx=0.5, rely=0.45, anchor="center", width=300, height=300)

# Bind mouse events to canvas
canvas_draw.bind("<Button-1>", get_x_and_y)
canvas_draw.bind("<B1-Motion>", draw_smth_tab2)

nav_frame = Frame(drawing_tab)
nav_frame.pack(side="bottom", pady=5)

# Create a button to save the image
save_button = Button(nav_frame, text="Save Image", command=save_image)
save_button.pack(side="left")

clear_button = Button(nav_frame, text="Clear all", command=clear_canvas_tab2)
clear_button.pack(side="left", padx=10)

# Start the main loop
root.mainloop()
