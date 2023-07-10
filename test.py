import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


symbol_dict = {'0':'α',
	       '1':'β',
		   '2':'γ',
		   '3':'δ',
		   '4':'λ',
		   '5':'μ',
		   '6':'Ω',
		   '7':'π',
		   '8':'φ',
			'9':'θ'}

def symbol(ind):
    symbols = ['α',
	       'β',
		   'γ',
		   'δ',
		   'λ',
		   'μ',
		   'Ω',
		   'π',
		   'φ',
			'θ']
    symb = symbols[ind.argmax()]
    return symb


# Load trained classifier from file
with open('MNIST_SVM.pickle', 'rb') as f:
    clf = pickle.load(f)


# Load image
img = Image.open("1.png") # convert to grayscale
img = img.resize((25, 25)) # resize to 25x25x3


# Convert image to numpy array
img_array = np.array(img)

img_flat = img_array.flatten()

digit_pred = clf.predict([img_flat])



print("Predicted Digit:", symbol_dict[str(digit_pred[0])])
plt.imshow(img)
plt.show()




'''
import torch
from PIL import Image
import numpy as np

# Predict sử dụng model đã train
def plot(data, model):
  data = torch.unsqueeze(data, dim=0) # unsqueeze data
  data = data.to(device)
  output = model(data)
  output = F.log_softmax(output, dim=1) # log softmax, chú ý dim
  pred = output.argmax(dim=1, keepdim=True) # argmax, chú ý keepdim, dim=1
  print("Predict Number : ", symbol_dict[str(pred[0][0].detach().cpu().numpy())])
  plt.imshow(data[0][0].detach().cpu().numpy(), cmap='gray')
  plt.show()

# Load the PyTorch model from the .pth file
model_1 = torch.load('model.pth')

# Load the image and convert it to a grayscale numpy array
img = Image.open("/content/1e1 (1).png")
img_array = np.array(img)
img_array = img_array.astype(np.float32)
img_array = torch.from_numpy(img_array)
data = img_array.permute(2, 0, 1)


'''

new_model = tf.keras.models.load_model('/content/my_model.h5')
new_model.summary()

def prediction(image_path,model):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap = 'gray')
    img = cv.resize(img,(25, 25))
    norm_image = cv.normalize(img, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])
    pred = model.predict([case])

    return 'Prediction: ' + symbol(pred)

prediction(model=new_model,)