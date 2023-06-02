import pickle
from PIL import Image
import numpy as np


# Load trained classifier from file
with open('MNIST_SVM.pickle', 'rb') as f:
    clf = pickle.load(f)


# Load image
img = Image.open("path_to_image").convert("L") # convert to grayscale
img = img.resize((28, 28), Image.ANTIALIAS) # resize to 28x28


# Convert image to numpy array
img_array = np.array(img)

img_flat = img_array.flatten()

digit_pred = clf.predict([img_flat])
print("Predicted Digit:", digit_pred[0])
