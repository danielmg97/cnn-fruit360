import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "fruits-360/Test/"

CATEGORIES = ["Pear Williams","Pepino"]

##--Half the categories in the dataset-- (uncomment at your own risk)
#CATEGORIES = ["Apple Braeburn","Apple Crimson Snow","Apple Golden 1","Apple Golden 2","Apple Golden 3","Apple Granny Smith","Apple Pink Lady","Apple Red 1","Apple Red 2","Apple Red 3","Apple Red Delicious","Apple Red Yellow 1","Apple Red Yellow 2","Apricot","Avocado","Avocado ripe","Banana","Banana Lady Finger","Banana Red","Cactus fruit","Cantaloupe 1","Cantaloupe 2","Carambula","Cherry 1","Cherry 2","Cherry Rainier","Cherry Wax Black","Cherry Wax Red","Cherry Wax Yellow","Chestnut","Clementine","Cocos","Dates","Granadilla","Grape Blue","Grape Pink","Grape White","Grape White 2","Grape White 3"]

##--All the categories in the dataset--  (uncomment at your own risk)
#CATEGORIES = ["Apple Braeburn","Apple Crimson Snow","Apple Golden 1","Apple Golden 2","Apple Golden 3","Apple Granny Smith","Apple Pink Lady","Apple Red 1","Apple Red 2","Apple Red 3","Apple Red Delicious","Apple Red Yellow 1","Apple Red Yellow 2","Apricot","Avocado","Avocado ripe","Banana","Banana Lady Finger","Banana Red","Cactus fruit","Cantaloupe 1","Cantaloupe 2","Carambula","Cherry 1","Cherry 2","Cherry Rainier","Cherry Wax Black","Cherry Wax Red","Cherry Wax Yellow","Chestnut","Clementine","Cocos","Dates","Granadilla","Grape Blue","Grape Pink","Grape White","Grape White 2","Grape White 3","Grape White 4","Grapefruit Pink","Grapefruit White","Guava","Hazelnut","Huckleberry","Kaki","Kiwi","Kohlrabi","Kumquats","Lemon","Lemon Meyer","Limes","Lychee","Mandarine","Mango","Mangostan","Maracuja","Melon Piel de Sapo","Mulberry","Nectarine","Orange","Papaya","Passion Fruit","Peach","Peach 2","Peach Flat","Pear","Pear Abate","Pear Kaiser","Pear Monster","Pear Red","Pear Williams","Pepino","Pepper Green","Pepper Red","Pepper Yellow","Physalis","Physalis with Husk","Pineapple","Pineapple Mini","Pitahaya Red","Plum","Plum 2","Plum 3","Pomegranate","Pomelo Sweetie","Quince","Rambutan","Raspberry","Redcurrant","Salak","Strawberry","Strawberry Wedge","Tamarillo","Tangelo","Tomato 1","Tomato 2","Tomato 3","Tomato 4","Tomato Cherry Red","Tomato Maroon","Tomato Yellow","Walnut"]

IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

import random

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "fruit360-CNN"

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=10,
          validation_split=0.3,
          callbacks=[tensorboard])
