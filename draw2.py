from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFilter

# Global variables
last_x, last_y = None, None
image_size = (25, 25)
drawing_points = []

def get_x_and_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw_smth(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill='black', width=4)
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

# Create the main window
app = Tk()
app.geometry("300x300")

# Create a canvas for drawing
canvas = Canvas(app, bg="white")
canvas.place(x=0, y=0, width=300, height=300)

# Bind mouse events to canvas
canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

# Create a button to save the image
save_button = Button(app, text="Save Image", command=save_image)
save_button.pack(side="bottom", pady=10)

# Start the main loop
app.mainloop()
