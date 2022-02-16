#MBY

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.datasets import fashion_mnist
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import History 
from tensorflow.keras.optimizers import SGD
import pandas as pd


mnist_data_train = pd.read_csv('sign_mnist_train.csv')
mnist_data_test = pd.read_csv('sign_mnist_test.csv')

print (mnist_data_test.shape)



images_train = mnist_data_train.iloc[:,1:]
labels_train = mnist_data_train.iloc[:,0]
images_test = mnist_data_test.iloc[:,1:]
labels_test = mnist_data_test.iloc[:,0]



#print("x_train shape:", X_train.shape, X_val.shape, "y_train shape:", y_train.shape, y_val.shape)
#print (X_train.shape[0], 'shape')


#plt.imshow(X_train[9456],cmap=plt.get_cmap('gray_r'))
#print (y_train[9456], 'y train')
#plt.show()


image_train_reshaped = images_train.values.reshape(-1,28,28,1)
image_test_reshaped = images_test.values.reshape(-1,28,28,1)

print (image_test_reshaped.shape, 'reshaped')
#plt.show()


image_train_reshaped = image_train_reshaped/255
image_test_reshaped = image_test_reshaped/255



max_value_train = labels_train.max(0)
max_value_test = labels_test.max(0)

print (max_value_train, max_value_test)
if max_value_train != max_value_test:
    print ('wtf')

#encode
labels_train_encoded = keras.utils.np_utils.to_categorical(labels_train, 25)
#print (labels_train_encoded, 'one hot')
labels_test_encoded = keras.utils.np_utils.to_categorical(labels_test, 25)

#TODO: replace with proper names
CLASS_NAMES = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8', 'c_9', 'c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16',
               'c_17', 'c_18', 'c_19', 'c_20', 'c_21', 'c_22', 'c_23', 'c_24', 'c_25']



model = Sequential() 

model.add(Conv2D(256 , (9,9) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((4,4) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (9,9) , strides = 1 , padding = 'same' , activation = 'relu'))
#Dropout
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((4,4) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (9,9) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((4,4) , strides = 2 , padding = 'same'))
model.add(Flatten())


model.add(Dense(25, activation='softmax'))

model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],)
model.summary()

history = model.fit(image_train_reshaped, labels_train_encoded, validation_data=(image_test_reshaped, labels_test_encoded), batch_size = 128, epochs = 10)



#loss, accuracy = model.evaluate(X_test, y_test)

print (history.history.keys())
plt.figure(figsize = (10,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.figure(figsize = (10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

labels_predictions = model.predict(image_test_reshaped)
labels_predictions = [np.argmax(pred) for pred in labels_predictions]

cm = confusion_matrix(labels_test, labels_predictions)
plt.figure(figsize = (10,6))
plt.show()
sns.heatmap(cm, annot=True)

print(classification_report(labels_test, labels_predictions, target_names = CLASS_NAMES))