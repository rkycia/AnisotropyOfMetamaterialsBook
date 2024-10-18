#! /usr/bin/env python

import numpy as np
import h5py
import pylab as plt
import cv2


#read X data - images
print("reading X data")


x = []
for fileName in ['./camelyonpatch_level_2_split_valid_x.h5', './camelyonpatch_level_2_split_test_x.h5']:
    print("... reading file: ", fileName)
    fx = h5py.File(fileName, 'r')

    for idx,image in enumerate(fx['x']):
        #imageRescaled = cv2.resize(image, (80,80))
    
    
        #equalization
        #src = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #h,s,v = cv2.split(src)
        #v = cv2.equalizeHist(v)
        #src = cv2.merge([h,s,v])
        #image = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
    
    
        #32 pixels around center containes at least one pixel of cancer cell
        imageRescaled = image[int(image.shape[0]/2.-20):int(image.shape[0]/2.+20), int(image.shape[1]/2.-20) : int(image.shape[1]/2.+20) ]
        imageRescaled = cv2.resize(imageRescaled, (40,40))
        #save image for later processing
        x.append(imageRescaled)
        #save samples to files
        #cv2.imwrite("image/{}.png".format(idx), image)
        #plt.imshow(image)
        #plt.show()

x = np.array(x)

print("DONE")
    

#preparing y data - cancer/no cancer
print("reading Y data")
y = []
for fileName in ['./camelyonpatch_level_2_split_valid_y.h5', './camelyonpatch_level_2_split_test_y.h5']:
    
    fy = h5py.File(fileName, 'r')
    
    for idx,datay in enumerate(fy['y']):
        y.append(datay.flatten())


        
y = np.array(y)
print("DONE")

print("dataset shape X: ", x.shape)
print("dataset shape Y: ", y.shape)
print("cancer cases: ", np.sum(y))


#save images for cancer
positive = x[y.reshape(-1)==1]
for idx,image in enumerate(positive):
   pass
   #cv2.imwrite("positive/{}.png".format(idx), image)


#shufle data randomly
from sklearn.utils import shuffle
x, y = shuffle(x, y)


#split into train, test and validation data
from sklearn.model_selection import train_test_split

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 



#create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.layers import Dropout, BatchNormalization, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation

import tensorflow as tf
 
model = Sequential([Input(shape=x.shape[1:]), 
                    RandomFlip("horizontal_and_vertical"),
                    RandomRotation(0.5),
                    Flatten(), 
                    #Dense(256, activation='relu'),
                    Dense(40, activation='relu'),
                    BatchNormalization(),
                    #Dense(256, activation='relu'),
                    Dense(1000, activation='relu'),
                    Dropout(0.2),
                    BatchNormalization(),
                    Dense(1, activation='sigmoid')
                    ])


#compile model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

#train model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val))

#accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig("output/accuracy.png")
plt.show()
# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.savefig("output/loss.png")
plt.show()


#test
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

#save full trained model
model.save('model/NNModel.h5')

#load model and test
new_model = tf.keras.models.load_model('model/NNModel.h5')
results = new_model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

