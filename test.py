import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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
img_array = img_array.astype(np.float32) /255.0
img_array = torch.from_numpy(img_array)
data = img_array.permute(2, 0, 1)


'''