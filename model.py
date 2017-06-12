from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(60, 60, 1)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(.25))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))\
model.add(MaxPooling2D(2,2))
model.add(Dropout(.25))

model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(Conv2D(128, 3, 3, activation='relu'))\
model.add(MaxPooling2D(2,2))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy','recall','precision'])

model.fit(X_train, Y_train, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
