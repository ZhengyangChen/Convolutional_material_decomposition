import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from keras.models import model_from_json
from keras.utils.visualize_util import plot

model = model_from_json(open(u'vgg_model_architecture.json').read())
model.load_weights(u'vgg_model_weights.h5')

X = sio.loadmat(u'../X_test')['X_test']
Y = sio.loadmat(u'../Y_test')['Y_test']
ans1 = []
ans2 = []
nb_classes = 5
nb_energy = 5

for N in range(15):
    X_eva = np.zeros([128*128, 32, 32, nb_energy])
    Y_eva = np.zeros([128*128, nb_classes])

    map = np.zeros([159, 159, nb_energy])
    map[16:144, 16:144, :] = X[N,:,:,:]

    for xx in range(128):
        for yy in range(128):
            X_eva[xx*128+yy,:,:,:] = map[xx:xx+32,yy:yy+32,:]
            Y_eva[xx*128+yy,:] = Y[N,xx,yy,:]

    X_eva = X_eva.astype('float32')

    xxx = X_eva[0:128*128]
    y1 = Y_eva[0:128*128]
    yy1 = np.zeros([128, 128, nb_classes])

    y2 = model.predict(xxx)
    yy2 = np.zeros([128, 128, nb_classes])

    for i in range(128):
        for j in range(128):
            yy1[i,j,:] = y1[i*128+j,:]
            yy2[i,j,:] = y2[i*128+j,:]

    ans1.append(yy1)
    ans2.append(yy2)

sio.savemat(u'vgg_result.mat', {'Y_label':ans1,'Y_pred':ans2})
