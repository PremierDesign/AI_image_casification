import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])


training_images = training_images[:4000]
training_labels = training_labels[:4000]
testing_images = testing_images[:800]
testing_labels = testing_labels[:800]


model = models.load_model('model.karas')

img = cv.imread('bird.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)
print(f'Prediction: {class_names[index]}')

