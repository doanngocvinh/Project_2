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

'''
import torch
from PIL import Image
import numpy as np

# Load the PyTorch model from the .pth file
model_1 = torch.load('model.pth')

# Load the image and convert it to a grayscale numpy array
img = Image.open("/content/gamma_93081.jpg").convert("L")
img = img.resize((28, 28), Image.ANTIALIAS) 
img_array = np.array(img)

# Normalize the pixel values to be between 0 and 1
img_array = img_array.astype(np.float32) /255.0

# Convert the numpy array to a PyTorch tensor
img_tensor = torch.from_numpy(img_array)

# Reshape the PyTorch tensor to match the expected input shape of the model
img_tensor = img_tensor.reshape((1, 1, 28, 28))

# Make a using the PyTorch model
with torch.no_grad():
    output = model(img_tensor)
    
# Get the predicted digit by selecting the class with the highest output probability
predicted_digit = torch.argmax(output).item()

print("Predicted Digit:", predicted_digit)
'''