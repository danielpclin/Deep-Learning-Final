import os

import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


def train(batch_size=500):
    checkpoint_path = 'checkpoint'
    log_dir = 'logs/1'
    epochs = 100
    # batch_size = 500
    img_width = 200
    img_height = 60
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv('train/data01_train.csv', delimiter=',')
    df['code'] = df['code'].apply(lambda el: list(el))
    df[[f'code{i}' for i in range(1, 7)]] = pd.DataFrame(df['code'].to_list(), index=df.index)
    for i in range(1, 7):
        df[f'code{i}'] = df[f'code{i}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)
    train_generator = datagen.flow_from_dataframe(dataframe=df, directory="train/data01_train", subset='training',
                                                  x_col="filename", y_col=[f'code{i}' for i in range(1, 7)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    valid_generator = datagen.flow_from_dataframe(dataframe=df, directory="train/data01_train", subset='validation',
                                                  x_col="filename", y_col=[f'code{i}' for i in range(1, 7)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    input_shape = (img_height, img_width, 3)
    main_input = Input(shape=input_shape)
    x = main_input
    # x = Conv2D(filters=32,
    #            kernel_size=(3, 3),
    #            padding='same',
    #            activation='relu')(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=64,
    #            kernel_size=(3, 3),
    #            padding='same',
    #            activation='relu')(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=128,
    #            kernel_size=(3, 3),
    #            padding='same',
    #            activation='relu')(x)
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=128,
    #            kernel_size=(3, 3),
    #            padding='same',
    #            activation='relu')(x)
    # x = Conv2D(filters=128,
    #            kernel_size=(3, 3),
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)
    # x = Conv2D(filters=256,
    #            kernel_size=(3, 3),
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=(0, 0))(x)
    x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = [Dense(len(alphabet), name=f'digit{i+1}', activation='softmax')(x) for i in range(6)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_digit6_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_digit6_accuracy', patience=10, verbose=1, mode='auto')
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorBoard, earlystop, checkpoint]
    # callbacks_list = [tensorBoard]

    model.summary()
    train_history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.n//valid_generator.batch_size,
        verbose=2,
        callbacks=callbacks_list
    )


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
