import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=gpu, floatX=float32"

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization



(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train.shape

num_classes = 10

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
input_shape = (28, 28, 1)

model=Sequential()
model.add(Conv2D(32,kernel_size=(5,5), strides=1,input_shape=input_shape))

model.add(Activation('relu'))

model.add(Conv2D(32,(5,5),strides=1,use_bias=False))

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3,3), strides=1,input_shape=input_shape))

model.add(Activation('relu'))


model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(128,use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(84,use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model_info = model.fit(x_train, y_train, batch_size=64, \
                         nb_epoch=30, verbose=0, validation_split=0.2)
print ( "Model başarıyla eğitildi" )

model.save('mnist_model2.h5')
print('Model Kaydedildi')

print ( "Model başarıyla eğitildi" )

