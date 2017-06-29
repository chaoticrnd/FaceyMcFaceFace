from keras.models import Sequential, model_from_json
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame, Series
import random
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')

with open('../experiments/models/model.json') as f:
    model = model_from_json(f.read())
model.load_weights('../experiments/weights/weights.h5')

X_train_fname = '../data/X_train.npy'
Y_train_fname = '../data/Y_train.npy'
X_train = np.load(X_train_fname)
Y_train = np.load(Y_train_fname)
X_cv_fname = '../data/X_cv.npy'
Y_cv_fname = '../data/Y_cv.npy'
X_cv = np.load(X_cv_fname)
Y_cv = np.load(Y_cv_fname)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
print(model.metrics_names)
score = model.evaluate(X_cv, Y_cv, verbose=1)
print(score)
