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

def genData(num, row, col, Xset, Yset):
    X = np.zeros([num, row, col, nb_energy])
    Y = np.zeros([num, nb_classes])
    N = Xset.shape[0]
    sizeX = Xset.shape[1]
    sizeY = Xset.shape[2]
    K = Xset.shape[3]
    map = np.zeros([N, sizeX+row-1, sizeY+col-1, K])
    for i in range(N):
        map[i,int(round(row/2)):int(round(row/2+sizeX)),int(round(col/2)):int(round(col/2+sizeY)),:] = Xset[i,:,:,:]
    for i in range(num):
        n = randint(0, N-1)
        xx = randint(0, sizeX-1)
        yy = randint(0, sizeY-1)
        ll = sample(range(K), nb_energy)
        for l in range(nb_energy):
            X[i,:,:,l] = map[n, xx:xx+row, yy:yy+col, ll[l]]
        Y[i,:] = Yset[n,xx,yy,:]
    X = X.astype('float32')
    Y = Y.astype('float32')
    return X, Y

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

X = np.zeros([100,128,128,5])
X[0:50,:,:,:] = sio.loadmat(u'../X_train1')['X_train']
X[50:100,:,:,:] = sio.loadmat(u'../X_train2')['X_train']
Y = sio.loadmat(u'../Y_train')['Y_train']

batch_size = 128
nb_classes = 5
nb_energy = 5
nb_epoch = 1
train_num = 64*128
test_num = 128*128
img_rows, img_cols = 32, 32
LR = 0.1

model = modelVGG()

plot(model, to_file=u'vgg_model.png',show_shapes=True)
sgd = SGD(lr=LR, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

X_train_set = X[0:85,:,:,:]
Y_train_set = Y[0:85,:,:,:]
X_val_set = X[85:100,:,:,:]
Y_val_set = Y[85:100,:,:,:]

last_loss = 100000
patience = 3.5
counter = 0
X_val, Y_val = genData(test_num, img_rows, img_cols, X_val_set, Y_val_set)
for o in range(2048):
    if counter >= patience:
        LR = LR * 0.5
        if LR < 0.0001:
            break
        sgd = SGD(lr=LR, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        counter = 0
        patience += 0.5
        X_val, Y_val = genData(test_num, img_rows, img_cols, X_val_set, Y_val_set)
        last_loss = 100000
    print(o, LR)
    X_train, Y_train = genData(train_num, img_rows, img_cols, X_train_set, Y_train_set)
    train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_val, Y_val))
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
