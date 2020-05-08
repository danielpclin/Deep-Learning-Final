import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)

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


def train():
    epochs = 50
    batch_size = 50
    img_width = 200
    img_height = 60
    train_dir = 'train/data01_train'
    test_dir = 'train/data01_train'
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    df = pd.read_csv('train/data01_train.csv', delimiter=',')
    df['code'] = df['code'].apply(lambda el: list(el))
    print(df)
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = datagen.flow_from_dataframe(dataframe=df, directory="train/data01_train", subset='training',
                                                  x_col="filename", y_col="code",
                                                  class_mode="categorical", classes=alphabet,
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    valid_generator = datagen.flow_from_dataframe(dataframe=df, directory="train/data01_train", subset='validation',
                                                  x_col="filename", y_col="code",
                                                  class_mode="categorical", classes=alphabet,
                                                  target_size=(img_height, img_width), batch_size=batch_size)
    # print(next(train_generator))
    # exit()
    # train_num = len(os.listdir(train_dir))
    # test_num = len(os.listdir(test_dir))
    input_shape = (img_height, img_width, 3)
    checkpoint_path = 'checkpoint'
    log_dir = 'logs/run21'
    main_input = Input(shape=input_shape)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(main_input)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.2)(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same',
               activation='relu')(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation='relu')(x)
    # x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
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
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # x = Dropout(0.2)(x)
    out = [Dense(36, name=f'digit{i+1}', activation='softmax')(x) for i in range(6)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_digit1_acc', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_digit1_acc', patience=5, verbose=1, mode='auto')
    tensorBoard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorBoard, earlystop, checkpoint]

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
    # a = generator('train/data01_train', 1)
    # print(next(a)[0][0])
    train()
