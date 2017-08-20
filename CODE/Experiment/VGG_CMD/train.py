import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization
from keras.layers.merge import add
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import h5py
from keras.models import model_from_json
from keras.utils.visualize_util import plot
from random import randint
from random import sample
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

def modelVGG():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, nb_energy)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

X_train_set = np.zeros([36324,32,32,5])
X_train_set[0:8000,:,:,:] = sio.loadmat(u'../X_train1')['X_train']
X_train_set[8000:16000,:,:,:] = sio.loadmat(u'../X_train2')['X_train']
X_train_set[16000:24000,:,:,:] = sio.loadmat(u'../X_train3')['X_train']
X_train_set[24000:32000,:,:,:] = sio.loadmat(u'../X_train4')['X_train']
X_train_set[32000:36324,:,:,:] = sio.loadmat(u'../X_train5')['X_train']
Y_train_set = sio.loadmat(u'../Y_train')['Y_train']
X_val_set = sio.loadmat(u'../X_val')['X_val']
Y_val_set = sio.loadmat(u'../Y_val')['Y_val']

batch_size = 128
nb_classes = 4
nb_energy = 5
nb_epoch = 1
train_num = 64*128
test_num = X_val_set.shape[0]
img_rows, img_cols = 32, 32
LR = 0.01

model = modelVGG()

plot(model, to_file=u'vgg_model.png',show_shapes=True)
sgd = SGD(lr=LR, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

X_train = np.zeros([train_num,32,32,5])
Y_train = np.zeros([train_num,4])

last_loss = 100000
patience = 2.5
counter = 0

for o in range(2048):
    if counter >= patience:
        LR = LR * 0.5
        if LR < 0.00001:
            break
        sgd = SGD(lr=LR, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        counter = 0
        patience += 0.5
        last_loss = 100000
    ll = sample(range(X_train_set.shape[0]),train_num)
    for i in range(train_num):
        X_train[i,:,:,:] = X_train_set[ll[i],:,:,:]
        for dir in range(randint(0, 1)):
            for k in range(nb_energy):
                X_train[i,:,:,k] = np.transpose(X_train[i,:,::-1,k])
        for ud in range(randint(0, 1)):
            for k in range(nb_energy):
                X_train[i,:,:,k] = X_train[i,::-1,:,k]
        for rl in range(randint(0, 1)):
            for k in range(nb_energy):
                X_train[i,:,:,k] = X_train[i,:,::-1,k]
        Y_train[i,:] = Y_train_set[ll[i],:]
    print(o, LR)
    train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_val_set, Y_val_set))
    val_loss =  min(train.history['val_loss'])
    if val_loss > last_loss:
        last_loss = val_loss
        counter += 1
    else:
        last_loss = val_loss
        counter = 0

json_string = model.to_json()
open(u'vgg_model_architecture.json','w').write(json_string)
model.save_weights(u'vgg_model_weights.h5')
