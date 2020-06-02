import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from functools import reduce
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model, models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


def main():
    if os.environ.get("LOCAL") == "TRUE":
        CUDA = 0
        if CUDA >= 0:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Restrict TensorFlow to only allocate 2GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA)
        # predict(batch_size=10, n=(54,), data=1) #
        # predict(batch_size=50, n=(0, 22, 23), data=1) # 96.06
        # predict(batch_size=50, n=(0, 22, 23, 30, 34, 40), data=1, method="occur_max") # 97.19
        # predict(batch_size=50, n=(0, 22, 23, 30, 34, 40, 42, 49, 54, 55), data=1, method="max") # 97.57
        # predict(batch_size=50, n=(0, 22, 23, 30, 34, 40, 42, 49, 54, 55), data=1, method="occur_max") # 97.76
        # predict(batch_size=50, n=(14, 21, 16, 30), data=2, method="max") # 92.10
        # predict(batch_size=50, n=(14, 21, 16, 30, 37, 45, 48, 50), data=2, method="max") # 94.06

    else:
        pass
        # predict(n=(0, 22, 23), data=1)
        # predict(batch_size=500, n=(0, 22, 23, 30, 34, 40, 42, 49, 54, 55), data=1, method="occur_max") # 97.76
        # predict(batch_size=500, n=(14, 21, 16, 30, 40, 37, 45, 48, 49, 50), data=2, method="max") # 94.26
        predict(batch_size=500, n=(42, 55, 54, 49, 57, 60, 34, 47, 59, 40, 48), data=1, method="occur_max")
        # predict(batch_size=500, n=(45, 48, 37, 50, 30, 40, 49, 34, 42, 39, 44), data=2, method="max")



def predict(batch_size=500, n=(0, 22, 23), data=1, method="ocuur_sum_max"):
    dataset = f"dev/data0{data}_dev"
    img_width = 200
    img_height = 60
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    df = pd.read_csv(f'{dataset}.csv', delimiter=',')
    pred = []
    for i in n:
        version = f"data0{data}_{i}"
        checkpoint_path = f'checkpoint_{version}.hdf5'
        datagen = ImageDataGenerator(rescale=1. / 255)
        predict_generator = datagen.flow_from_dataframe(dataframe=df, directory=dataset,
                                                        x_col="filename", class_mode=None, shuffle=False,
                                                        target_size=(img_height, img_width), batch_size=batch_size)
        model = models.load_model(checkpoint_path)
        _pred = model.predict_generator(
            predict_generator,
            steps=predict_generator.n // predict_generator.batch_size,
            verbose=1,
        )
        pred.append(_pred)
        K.clear_session()

    if len(n) == 1:
        pred = pred[0]
        result = ["" for _ in range(len(pred[0]))]
        pred = np.argmax(pred, axis=2)
        for digit in pred:
            for index, code in enumerate(digit):
                result[index] = result[index] + int_to_char[code % len(alphabet)]
        df['code'] = result
    else:
        pred_sum_argmax = np.argmax(reduce(np.add, pred), axis=2)
        pred_argmax_concat = np.concatenate(np.expand_dims(np.argmax(pred, axis=3), axis=3), axis=2)
        pred_concat_argmax = np.argmax(np.concatenate(pred, axis=2), axis=2)
        result = ["" for _ in range(len(pred_argmax_concat[0]))]
        if method == "occur_max":
            for index_digit, digit in enumerate(pred_argmax_concat):
                for index_code, codes in enumerate(digit):
                    (values, counts) = np.unique(codes, return_counts=True)
                    code = values[counts == counts.max()]
                    # print(code)
                    if len(code) > 1:
                        result[index_code] = result[index_code] + int_to_char[pred_concat_argmax[index_digit][index_code] % len(alphabet)]
                    else:
                        result[index_code] = result[index_code] + int_to_char[code[0] % len(alphabet)]
        elif method == "occur_sum_max":
            for index_digit, digit in enumerate(pred_argmax_concat):
                for index_code, codes in enumerate(digit):
                    (values, counts) = np.unique(codes, return_counts=True)
                    code = values[counts == counts.max()]
                    # print(code)
                    if len(code) > 1:
                        result[index_code] = result[index_code] + int_to_char[pred_sum_argmax[index_digit][index_code] % len(alphabet)]
                    else:
                        result[index_code] = result[index_code] + int_to_char[code[0] % len(alphabet)]
        elif method == "max":
            for digit in pred_concat_argmax:
                for index, code in enumerate(digit):
                    result[index] = result[index] + int_to_char[code % len(alphabet)]
        elif method == "sum_max":
            for digit in pred_sum_argmax:
                for index, code in enumerate(digit):
                    result[index] = result[index] + int_to_char[code % len(alphabet)]
    df['code'] = result
    if len(n) == 1:
        df.to_csv(f'predict/data0{data}_{n[0]}.csv', index=False)
    else:
        df.to_csv(f'predict/data0{data}_{"_".join(map(str, n))}_{method}.csv', index=False)
    print(df)


if __name__ == "__main__":
    main()
