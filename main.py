import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam

# os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)


def main():
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
        for i in range(1002, 1011):
            train(50, n=1001, data=2)
    else:
        for i in range(150, 171):
            train(n=i, data=2)

def Conv2d_BN(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu')(x)
        return x
    return block


# Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(filters, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    def block(input_x):
        # x = Conv2d_BN(filters=filters/4, kernel_size=(1, 1), strides=strides, padding='same')(input_x)
        # x = Conv2d_BN(filters=filters/4, kernel_size=kernel_size, padding='same')(x)
        # x = Conv2d_BN(filters=filters, kernel_size=(1, 1), padding='same')(x)
        x = Conv2d_BN(filters=filters, kernel_size=(3, 3), padding='same')(input_x)
        x = Conv2d_BN(filters=filters, kernel_size=(3, 3), padding='same')(x)
        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = Conv2d_BN(filters=filters, strides=strides, kernel_size=kernel_size)(input_x)
            x = Add()([x, shortcut])
        else:
            x = Add()([x, input_x])
        return x
    return block


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


class MinimumEpochEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False, min_epoch=30):
        super(MinimumEpochEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


def train(batch_size=500, n=50, data=1):
    dataset = f"train/data0{data}_train"
    version = f"data0{data}_{n}"
    checkpoint_path = f'checkpoint_{version}.hdf5'
    log_dir = f'logs/{version}'
    epochs = 100
    img_width = 200
    img_height = 60
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{dataset}.csv', delimiter=',')
    df['code'] = df['code'].apply(lambda el: list(el))
    df[[f'code{i}' for i in range(1, 7)]] = pd.DataFrame(df['code'].to_list(), index=df.index)
    for i in range(1, 7):
        df[f'code{i}'] = df[f'code{i}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=df, directory=dataset, subset='training',
                                                  x_col="filename", y_col=[f'code{i}' for i in range(1, 7)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    valid_generator = datagen.flow_from_dataframe(dataframe=df, directory=dataset, subset='validation',
                                                  x_col="filename", y_col=[f'code{i}' for i in range(1, 7)],
                                                  class_mode="multi_output",
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    input_shape = (img_height, img_width, 3)
    main_input = Input(shape=input_shape)
    x = main_input
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3), with_conv_shortcut=True)(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    out = [Dense(len(alphabet), name=f'digit{i + 1}', activation='softmax')(x) for i in range(6)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    if data == 1:
        earlystop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=5)
    else:
        earlystop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=10)
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto', min_lr=0.00005)
    callbacks_list = [tensorBoard, earlystop, checkpoint, reduceLR]

    model.summary()
    train_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.n // valid_generator.batch_size,
        verbose=1,
        callbacks=callbacks_list
    )
    with open(f"{version}.txt", "w") as file:
        loss_idx = np.argmin(train_history.history['val_loss'])
        digit6_idx = np.argmax(train_history.history['val_digit6_accuracy'])
        file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit1_accuracy'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit2_accuracy'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit3_accuracy'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit4_accuracy'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit5_accuracy'][loss_idx]}\n")
        file.write(f"{train_history.history['val_digit6_accuracy'][loss_idx]}\n")
        file.write(f"{'-'*20}\n")
        file.write(f"{train_history.history['val_loss'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit1_accuracy'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit2_accuracy'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit3_accuracy'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit4_accuracy'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit5_accuracy'][digit6_idx]}\n")
        file.write(f"{train_history.history['val_digit6_accuracy'][digit6_idx]}\n")
    K.clear_session()


if __name__ == "__main__":
    main()
