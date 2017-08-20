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
    K = nb_energy
    map = np.zeros([N, sizeX+row-1, sizeY+col-1, K])
    for i in range(N):
        map[i,int(round(row/2)):int(round(row/2+sizeX)),int(round(col/2)):int(round(col/2+sizeY)),0:nb_energy] = Xset[i,:,:,0:nb_energy]
    for i in range(num):
        n = randint(0, N-1)
        xx = randint(0, sizeX-1)
        yy = randint(0, sizeY-1)
        X[i,:,:,:] = map[n, xx:xx+row, yy:yy+col, :]
        Y[i,:] = Yset[n,xx,yy,:]
    X = X.astype('float32')
    Y = Y.astype('float32')
    return X, Y

def modelDRN():
    inputLayer = Input(shape=(img_rows, img_cols, nb_energy),name='input')
    c0 = Conv2D(64, (5, 5), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', name='c0')(inputLayer)
    b0 = BatchNormalization(axis=1, name='b0')(c0)
    a0 = Activation('relu', name='a0')(b0)

    r1_1 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r1_1')(a0)
    b1_1 = BatchNormalization(axis=1, name='b1_1')(r1_1)
    a1_1 = Activation('relu', name='a1_1')(b1_1)
    r1_2 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r1_2')(a1_1)
    b1_2 = BatchNormalization(axis=1, name='b1_2')(r1_2)
    m1 = add([a0, b1_2], name='m1')
    a1 = Activation('relu', name='a1')(m1)
    r2_1 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r2_1')(a1)
    b2_1 = BatchNormalization(axis=1, name='b2_1')(r2_1)
    a2_1 = Activation('relu', name='a2_1')(b2_1)
    r2_2 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r2_2')(a2_1)
    b2_2 = BatchNormalization(axis=1, name='b2_2')(r2_2)
    m2 = add([a1, b2_2], name='m2')
    a2 = Activation('relu', name='a2')(m2)

    r3_0 = Conv2D(64, (1, 1), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', strides=(2, 2), name='r3_0')(a2)
    b3_0 = BatchNormalization(axis=1, name='b3_0')(r3_0)
    a3_0 = Activation('relu', name='a3_0')(b3_0)
    r3_1 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', strides=(2, 2), name='r3_1')(a2)
    b3_1 = BatchNormalization(axis=1, name='b3_1')(r3_1)
    a3_1 = Activation('relu', name='a3_1')(b3_1)
    r3_2 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r3_2')(a3_1)
    b3_2 = BatchNormalization(axis=1, name='b3_2')(r3_2)
    m3 = add([b3_0, b3_2], name='m3')
    a3 = Activation('relu', name='a3')(m3)
    r4_1 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r4_1')(a3)
    b4_1 = BatchNormalization(axis=1, name='b4_1')(r4_1)
    a4_1 = Activation('relu', name='a4_1')(b4_1)
    r4_2 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r4_2')(a4_1)
    b4_2 = BatchNormalization(axis=1, name='b4_2')(r4_2)
    m4 = add([a3, b4_2], name='m4')
    a4 = Activation('relu', name='a4')(m4)
    r5_1 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r5_1')(a4)
    b5_1 = BatchNormalization(axis=1, name='b5_1')(r5_1)
    a5_1 = Activation('relu', name='a5_1')(b5_1)
    r5_2 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r5_2')(a5_1)
    b5_2 = BatchNormalization(axis=1, name='b5_2')(r5_2)
    m5 = add([a4, b5_2], name='m5')
    a5 = Activation('relu', name='a5')(m5)

    r6_0 = Conv2D(128, (1, 1), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', strides=(2, 2), name='r6_0')(a5)
    b6_0 = BatchNormalization(axis=1, name='b6_0')(r6_0)
    a6_0 = Activation('relu', name='a6_0')(b6_0)
    r6_1 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', strides=(2, 2), name='r6_1')(a5)
    b6_1 = BatchNormalization(axis=1, name='b6_1')(r6_1)
    a6_1 = Activation('relu', name='a6_1')(b6_1)
    r6_2 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r6_2')(a6_1)
    b6_2 = BatchNormalization(axis=1, name='b6_2')(r6_2)
    m6 = add([b6_0, b6_2], name='m6')
    a6 = Activation('relu', name='a6')(m6)
    r7_1 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r7_1')(a6)
    b7_1 = BatchNormalization(axis=1, name='b7_1')(r7_1)
    a7_1 = Activation('relu', name='a7_1')(b7_1)
    r7_2 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r7_2')(a7_1)
    b7_2 = BatchNormalization(axis=1, name='b7_2')(r7_2)
    m7 = add([a6, b7_2], name='m7')
    a7 = Activation('relu', name='a7')(m7)
    r8_1 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r8_1')(a7)
    b8_1 = BatchNormalization(axis=1, name='b8_1')(r8_1)
    a8_1 = Activation('relu', name='a8_1')(b8_1)
    r8_2 = Conv2D(128, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', padding='same', name='r8_2')(a8_1)
    b8_2 = BatchNormalization(axis=1, name='b8_2')(r8_2)
    m8 = add([a7, b8_2], name='m8')
    a8 = Activation('relu', name='a8')(m8)

    c9 = Conv2D(256, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', name='c9')(a8)
    b9 = BatchNormalization(axis=1, name='b9')(c9)
    a9 = Activation('relu', name='a9')(b9)
    c10 = Conv2D(256, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', name='c10')(a9)
    b10 = BatchNormalization(axis=1, name='b10')(c10)
    a10 = Activation('relu', name='a10')(b10)
    c11 = Conv2D(256, (3, 3), kernel_initializer='glorot_normal', bias_initializer='glorot_normal', name='c11')(a10)
    b11 = BatchNormalization(axis=1, name='b11')(c11)
    a11 = Activation('relu', name='a11')(b11)

    flattenLayer = Flatten(name='flatten')(a11)
    dense1 = Dense(64, activation='relu', name='dense1')(flattenLayer)
    dense2 = Dense(nb_classes, activation='softmax', name='dense2')(dense1)

    model = Model(inputs=inputLayer, outputs=dense2)
    return model

X = sio.loadmat(u'../X_train')['X_train']
Y = sio.loadmat(u'../Y_train')['Y_train']

batch_size = 128
nb_classes = 5
nb_energy = 5
nb_epoch = 1
train_num = 64*128
test_num = 128*128
img_rows, img_cols = 32, 32
LR = 0.1

model = modelDRN()

plot(model, to_file=u'drn_model.png',show_shapes=True)
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
open(u'drn_model_architecture.json','w').write(json_string)
model.save_weights(u'drn_model_weights.h5')
