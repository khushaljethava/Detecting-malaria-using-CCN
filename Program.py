# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:26:50 2020

@author: d3evil4
"""

#importing Libraries

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import ImageDataGenerator

# initialising the CNN

classifier = Sequential()


# Step 1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu' ))

#step 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding a second comvolution layer
classifier.add(Convolution2D(32,3,3,activation = 'relu' ))

classifier.add(MaxPooling2D(pool_size=(2,2)))



# step3 Flattening
classifier.add(Flatten())

#step4 Full connection
classifier.add(Dense(output_dim = 128,activation = 'relu'))

classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#Part 2 - Fitting the data

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'cell_images_small/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'cell_images_small/valid',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
        samples_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


# Part - Making new predictions
import numpy as np
import matplotlib.pyplot as plt 
from keras.preprocessing import image
test_image1 = image.load_img('cell_images_small/random1.png',target_size=(64,64))
test_image = image.img_to_array(test_image1)
test_image= np.expand_dims(test_image,axis=0)
classifier.predict(test_image)

result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] ==1:
    prediction = 'Not infected '
else:
    prediction = 'Infected'
    
print(prediction)
plt.imshow(test_image1)