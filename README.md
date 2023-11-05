# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![mnist](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/91f82cc2-b402-4385-9fed-59f31c5a72fc)


## Neural Network Model

![NNmodel](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/707da1ba-a4cd-416e-8b94-c1a480e41180)


## DESIGN STEPS
- STEP-1: Import tensorflow and preprocessing libraries
- STEP 2: Download and load the dataset
- STEP 3: Scale the dataset between it's min and max values
- STEP 4: Using one hot encode, encode the categorical values
- STEP-5: Split the data into train and test
- STEP-6: Build the convolutional neural network model
- STEP-7: Train the model with the training data
- STEP-8: Plot the performance plot
- STEP-9: Evaluate the model with the testing data
- STEP-10: Fit the model and predict the single input

## PROGRAM
```
program Developed by:Pavan Kishore M
Register Number : 212221230076
```

```python
#importing libraries
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
```

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
X_test.shape
```

```python
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
```

```python
y_train.shape
```

```python
#checking the grayscale value range of the images
print(X_train.min())
print(X_train.max())

#to bring value to 0-1 range
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

print(X_train_scaled.min())
print(X_train_scaled.max())
```

```python
y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

print(type(y_train_onehot))
y_train_onehot.shape
```

```python
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]
```

```python
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```

```python
#network model
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
```

```python
metrics = pd.DataFrame(model.history.history)
metrics.head()
```

```python
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
```

```python
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
```

```python
#prediction for single input- 1
img = image.load_img('numeg.PNG')
type(inv)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```

```python
#checking input 1 inverted
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

```python
#prediction for single input- 2
img1 = image.load_img('numeg2.PNG')
type(inv)
img1_tensor = tf.convert_to_tensor(np.asarray(img1))
img1_28 = tf.image.resize(img1_tensor,(28,28))
img1_28_gray = tf.image.rgb_to_grayscale(img1_28)
img1_28_gray_scaled = img1_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img1_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)

#checking input 2 inverted
img1_28_gray_inverted = 255.0-img1_28_gray
img1_28_gray_inverted_scaled = img1_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img1_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![](1.png)

![](2.png)

### Classification Report

![3](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/82bedd33-a379-434e-b9ef-01297650fcde)

### Confusion Matrix
![conmatrix](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/59266d3d-2252-4b80-a5b0-cccca1b04fc7)



### New Sample Data Prediction
### Sample image - 1 : input & output
![in11](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/d1460d4c-2069-40b0-9004-e3a8d5a55118)


![in12](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/3654be5a-8acb-471f-85c8-a7f567ee0787)




### Sample image - 2 : input & output

![in21](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/910e29ed-2483-44ea-811d-48ed9adcdcf4)

![in22](https://github.com/pavankishore-AIDS/mnist-classification/assets/94154941/acfb17da-9ef3-4668-98eb-1644eead7a45)



## RESULT

Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
