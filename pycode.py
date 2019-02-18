import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
from keras.layers import Convolution2D
import pandas as pd


classifier=Sequential()
classifier.add(Convolution2D(32,(4,4),input_shape=(128,128,3),activation="relu",padding="same"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,(4,4),activation="relu",padding="same"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,(4,4),activation="relu",padding="same"))
classifier.add(Flatten())
classifier.add(Dense(units=64,activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=10,activation="relu"))
classifier.add(Dense(units=10,activation="relu"))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=10,activation="relu"))
classifier.add(Dense(units=(1),activation="sigmoid"))
classifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/home/jainil/program/datasets/catdog/training_set',
        target_size=(128, 128),
        batch_size=100,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory('/home/jainil/program/datasets/catdog/test_set',
        target_size=(64, 64),
        batch_size=100,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=20)
    
a=image.load_img('/home/jainil/program/datasets/catdog/single_prediction/pic1.jpg',target_size=(128,128))
a=image.img_to_array(a)
a=np.expand_dims(a,axis=0)
print(a.shape)
b=classifier.predict(a)
if(b>0.5):
    print("it is a cat")
else:
    print("it is a dog")
