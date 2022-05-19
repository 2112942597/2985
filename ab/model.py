#  If you want to get the detailed code, please contact lwf@haut.edu.cn
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from pathlib import Path
import ast
from tqdm import tqdm_notebook, tqdm
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import  CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def dice(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection+1 ) / (keras.sum(y_true_f) + keras.sum(y_pred_f)+1 )

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

from tensorflow.keras.applications.efficientnet import EfficientNetB4
backbone = EfficientNetB4(weights='imagenet',
                            include_top=False,
                            input_shape=(256,256,3))
backbone.layers[326].output,backbone.layers[149].output,backbone.layers[90].output,backbone.layers[31].output

........

K.clear_session()
img_size = 256
model = U_Net(input_shape=(img_size,img_size,3),dropout_rate=0.25)

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights_ROI.best.hdf5".format('quan-HAUT_xia-jsrt-sanweixia_effi_')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-7)
early = EarlyStopping(monitor="val_dice",
                      mode="max",
                      patience=8) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]

from IPython.display import clear_output
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

model.compile(optimizer=Adam(lr=2e-4),
              loss=[dice_loss],
           metrics = [dice, 'binary_accuracy'])

#x_vol.shape,x_seg.shape,y_vol.shape,y_seg.shape
loss_history = model.fit(x = train_vol,
                       y = train_seg,
                         batch_size = 16,
                  epochs = 100,
                  validation_data =(validation_vol,validation_seg),
                  callbacks=callbacks_list)
