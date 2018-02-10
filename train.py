import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models.model import UNET
from metrics.mean_iou import mean_iou


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

model = UNET((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])


X_train = np.load('input/X_train.npy')
Y_train = np.load('input/Y_train.npy')


# Fit models
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('models/models-dsbowl2018-1.h5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, \
                                        patience=1, min_lr=0.001)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                    callbacks=[earlystopper, checkpointer, reduce_lr])