import os

import numpy as np
from PIL import Image
from keras import Input, Model
from keras import losses
from keras import metrics
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, \
    AveragePooling2D


def prepare_dataset(my_path):
    radiographies_pixels = []
    ok = 0
    for my_file in os.listdir(my_path):
        ok += 1
        if ok > 1000:
            my_radiography = Image.open(os.path.join(my_path, my_file), 'r')
            radiography = list(my_radiography.getdata())
            radiography = np.array(radiography)
            radiographies_pixels.append(radiography)

    radiographies_pixels = np.array(radiographies_pixels, dtype='uint8')
    return radiographies_pixels


def train_the_network(dicom_path, masks_path):
    dicom_photos = prepare_dataset(dicom_path)
    mask_photos = prepare_dataset(masks_path)
    return dicom_photos, mask_photos


def my_cnn_network(width, dicom_path, masks_path, prag, train_dim, test_dim):
    x_train, y_train = train_the_network(dicom_path, masks_path)
    x_train = x_train.reshape(train_dim, width, width, 1)
    y_train = y_train.reshape(train_dim, width, width, 1)

    # arhitectura
    inp = Input((width, width, 1))
    l = Conv2D(32, (2, 2), padding='same')(inp)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = AveragePooling2D((2, 2))(l)

    l = Conv2D(64, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = AveragePooling2D((2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = AveragePooling2D((2, 2))(l)

    l = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv2DTranspose(32, (3, 3), padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    l = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('softmax')(l)

    MyModel = Model(inp, decoded)

    MyModel.summary()
    print()

    MyModel.compile(optimizer='adagrad', loss=losses.mse,
                    metrics=['accuracy', metrics.Recall(), metrics.Precision(), metrics.TruePositives(),
                             metrics.TrueNegatives(), metrics.FalseNegatives(), metrics.FalsePositives()])
    MyModel.fit(x_train, y_train, batch_size=32, epochs=1000)
    MyModel.save('model.h5')


my_cnn_network(32, r'C:\Users\Lucia\Desktop\smallimgdataset\dicom', r'C:\Users\Lucia\Desktop\smallimgdataset\mask', 0.7,
               8871, 2218)
