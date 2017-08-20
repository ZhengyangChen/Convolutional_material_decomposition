import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from keras.models import model_from_json
from keras.utils.visualize_util import plot

model = model_from_json(open(u'vgg_model_architecture.json').read())
model.load_weights(u'vgg_model_weights.h5')

X_test = sio.loadmat(u'../X_test')['X_test']
Y_test = sio.loadmat(u'../Y_test')['Y_test']
ans1 = []
ans2 = []

X_test = X_test.astype('float32')
Y_test = Y_test.astype('float32')

Y_pred = model.predict(X_test)

ans1.append(Y_test)
ans2.append(Y_pred)

sio.savemat(u'vgg_result.mat', {'Y_label':ans1,'Y_pred':ans2})
