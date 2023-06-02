# Project_2

Predict a photo:
  pickle_in = open('model','rb') (sửa model thanh file pickle, e.g: MNIST_KNN.pickle)
  clf = pickle.load(pickle_in)
  y_pred = clf.predict(anh)   (sửa anh thành ảnh định predict, e.g 123.jpg) 
  print(y_pred)
