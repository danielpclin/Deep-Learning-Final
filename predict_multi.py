import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model, models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


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
        predict(batch_size=50, n=(0, 22, 23), data=1)
    else:
        predict(n=(0, 22, 23), data=1)


def predict(batch_size=500, n=(0, 22, 23), data=1):
    dataset = f"dev/data0{data}_dev"
    img_width = 200
    img_height = 60
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{dataset}.csv', delimiter=',')
    datagen = ImageDataGenerator(rescale=1. / 255)
    predict_generator = datagen.flow_from_dataframe(dataframe=df, directory=dataset,
                                                    x_col="filename", class_mode=None, shuffle=False,
                                                    target_size=(img_height, img_width), batch_size=batch_size)
    pred = []
    for i in n:
        version = f"data0{data}_{i}"
        checkpoint_path = f'checkpoint_{version}.hdf5'
        model = models.load_model(checkpoint_path)
        pred.append(model.predict_generator(
            predict_generator,
            steps=predict_generator.n // predict_generator.batch_size,
            verbose=1,
        ))
        K.clear_session()
    pred_concat = np.argmax(np.concatenate(pred, axis=2), axis=2)
    result = ["" for _ in range(len(pred_concat[0]))]
    for digit in pred_concat:
        for index, code in enumerate(digit):
            result[index] = result[index] + int_to_char[code % len(alphabet)]
    df['code'] = result
    df.to_csv(f'predict/data0{data}_{"_".join(str(n))}.csv', index=False)
    print(df)


if __name__ == "__main__":
    main()
