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
        # for i in range(1005, 1011):
        #     train(50, n=i, data=2, res=True)
        train(50, n=1005, data=2, res=True)
    else:
        # for i in range(159, 161):  # stable and performant (2/2)
        #     train(n=i, data=2, res=True)
        # for i in range(161, 166):  # unstable and performant < res (lr may need to decrease) (3/5)
        #     train(n=i, data=2, res=False)
        # for i in range(166, 171):  # unstable and performant <= res (lr may need to decrease) (2.5/5)
        #     train(n=i, data=2, res=False, quad=True)
        # for i in range(171, 176):  # unstable super performant (3/5)
        #     train(n=i, data=2, res=False, quad=False, drop=True)
        # for i in range(176, 181):  # stable? performant < res (lr may need to decrease) (3/3)
        #     train(n=i, data=2, res=False, quad=False, drop=True, convBLK=2)
        # for i in range(181, 186):  # stable 3/5 performant
        #     train(n=i, data=2, res=True)
        # for i in range(186, 191):  # stable 5/5 performant
        #     train(n=i, data=2, res=True, quad=True)
        # for i in range(191, 196):  # unstable 0/1 not performant
        #     train(n=i, data=2, res=False, quad=True, drop=True)
        # for i in range(196, 201):  #
        #     train(n=i, data=2, res=False, quad=True, drop=True, convBLK=1)
        # for i in range(201, 206):  #
        #     train(n=i, data=2, res=False, quad=False, drop=True, convBLK=2)
        # for i in range(206, 211):  #
        #     train(n=i, data=2, res=False, quad=True, drop=True, convBLK=2)
        # for i in range(211, 216):  #
        #     train(n=i, data=2, res=False, quad=True, drop=True, convBLK=1)
        # for i in range(217, 219):  #
        #     train(batch_size=500, n=i, data=2, res=False, quad=True, drop=True)
        # for i in range(219, 221):  #
        #     train(batch_size=500, n=i, data=2, res=False, quad=False, drop=True)
        # for i in range(221, 231):  #
        #     train(batch_size=500, n=i, data=2, res=True)
        # for i in range(231, 237):  # old
        #     train(batch_size=500, n=i, data=2, res=True, quad=True)
        # for i in range(244, 246):
        #     train(batch_size=500, n=i, data=2, res=False, conv="back")
        # for i in range(242, 244):
        #     train(batch_size=500, n=i, data=2, res=False, conv="front")
        # for i in range(237, 241):  #
        #     train(batch_size=500, n=i, data=2, res=True, quad=True)
        # for i in range(241, 242):
        #     train(batch_size=500, n=i, data=2, res=True, quad=False)
        # for i in range(246, 266):
        #     train(batch_size=500, n=i, data=2, res=True, quad=True)
        for i in range(266, 276):
            train(batch_size=500, n=i, data=2, res=False)

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


def Conv2D_Block(filters, kernel_size, strides=(1, 1), padding='same'):
    def block(input_x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input_x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return block


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


class MinimumEpochReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0, verbose=0, mode='auto', factor=0.1, cooldown=0, min_lr=0., min_epoch=30):
        super(MinimumEpochReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


def train(batch_size=500, n=1000, data=2, res=True, quad=True, conv="front"):
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
    if res:
        x = main_input
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
        x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
        x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
        if quad:
            x = Residual_Block(filters=64, kernel_size=(3, 3))(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.2)(x)
        x = Residual_Block(filters=128, kernel_size=(3, 3), with_conv_shortcut=True)(x)
        x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
        x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
        if quad:
            x = Residual_Block(filters=128, kernel_size=(3, 3))(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.2)(x)
        x = Residual_Block(filters=256, kernel_size=(3, 3), with_conv_shortcut=True)(x)
        x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
        x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
        if quad:
            x = Residual_Block(filters=256, kernel_size=(3, 3))(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.3)(x)
        x = Conv2D(filters=512, kernel_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
    else:
        x = main_input
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.4)(x)
    out = [Dense(len(alphabet), name=f'digit{i + 1}', activation='softmax')(x) for i in range(6)]
    model = Model(main_input, out)
    if res:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.005), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    if data == 1:
        earlystop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=5)
    else:
        earlystop = MinimumEpochEarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', min_epoch=20)
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    if res:
        reduceLR = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, mode='auto', min_lr=0.00001, min_epoch=15)
    else:
        reduceLR = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, cooldown=1, mode='auto', min_lr=0.00001, min_epoch=20)
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
