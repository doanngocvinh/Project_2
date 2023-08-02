import sys
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

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
    '9': 'θ',
    '10':'h',
    '11':'Y',
    '12':'đ'
}

mnist_data = np.load('../new_data/dataset.npz')
images = mnist_data['images']
labels = mnist_data['labels']
train_img, test_img, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)


'''data = MNIST('./MNIST_Dataset_Loader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)
'''

#Features
X = train_img.reshape(train_img.shape[0], -1)
test_img = test_img.reshape(test_img.shape[0], -1)

#Labels
y = train_labels


# Prepare Classifier Training and Testing Data
print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)


# Pickle the Classifier for Future Use
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train,y_train)

with open('MNIST_SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle','rb')
clf = pickle.load(pickle_in)

#print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

#print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

#print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)


print('\nSVM Trained Classifier Accuracy: ',acc)
print('\nAccuracy of Classifier on Validation Images: ',accuracy)


#print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

#print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
acc_test = accuracy_score(test_labels,test_labels_pred)

#print('\n Creating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

#print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('\nAccuracy of Classifier on Test Images: ',acc_test)

result = {
    'accuracy_validation': accuracy,
    'accuracy_test': acc_test,
}


# Ghi các giá trị vào file văn bản
with open('../GUI/svm_evalute.txt', 'w') as file:
	for key, value in result.items():
		file.write(f"{round(value,3)}\n")




'''# Show the Test Images with Original and Predicted Labels
a = np.random.randint(1,40,15)
for i in a:
	two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
	plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i],symbol_dict[str(test_labels_pred[i])]))
	plt.imshow(two_d, interpolation='nearest',cmap='gray')
	plt.show()
#---------------------- EOC ---------------------#'''

