import os

import pandas as pd
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


def train(batch_size=500):
    version = "conv256_double_0.4_7"
    checkpoint_path = f'checkpoint_{version}.hdf5'
    epochs = 100
    # batch_size = 500
    img_width = 200
    img_height = 60
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv('dev/data01_dev.csv', delimiter=',')
    # df['code'] = df['code'].apply(lambda el: list(el))
    # df[[f'code{i}' for i in range(1, 7)]] = pd.DataFrame(df['code'].to_list(), index=df.index)
    # for i in range(1, 7):
    #     df[f'code{i}'] = df[f'code{i}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
    datagen = ImageDataGenerator(rescale=1. / 255)
    predict_generator = datagen.flow_from_dataframe(dataframe=df, directory="dev/data01_dev",
                                                    x_col="filename", class_mode=None, shuffle=False,
                                                    target_size=(img_height, img_width), batch_size=batch_size)
    input_shape = (img_height, img_width, 3)
    main_input = Input(shape=input_shape)
    x = main_input
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               # padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               # padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               # padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               # padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = [Dense(len(alphabet), name=f'digit{i+1}', activation='softmax')(x) for i in range(6)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.load_weights(checkpoint_path)
    pred = model.predict_generator(
        predict_generator,
        steps=predict_generator.n // predict_generator.batch_size,
        verbose=1,
    )
    result = ["" for _ in range(len(pred[0]))]
    pred = np.argmax(pred, axis=2)
    for digit in pred:
        for index, code in enumerate(digit):
            result[index] = result[index] + int_to_char[code]
    df['code'] = result
    df.to_csv(f'predict/{version}.csv', index=False)
    print(df)


if __name__ == "__main__":
    if os.environ.get("LOCAL") == "TRUE":
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        train(50)
    else:
        train()
