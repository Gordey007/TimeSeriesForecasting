# univariate convlstm example

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError, Accuracy
from keras.optimizers import Adam
from keras.saving.save import load_model
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
# Графические пакеты
import matplotlib.pyplot as plt

plt.matplotlib.use('WebAgg')

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def normalize(data, data_mean=0, data_std=0, flag=0):
    if flag == 0:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)

    return (data - data_mean) / data_std, data_mean, data_std


def des_normalize(data, data_mean, data_std):
    return data * data_std + data_mean


def get_data(file, path, column_name, n_ahead, n_seq, n_steps, n_features):
    data_mean = 0
    data_std = 0

    # 158827 - 2019
    # 166799 - 2020
    # 175612 - 2021
    # 176596 - 2021-02-11
    # define input sequence

    start_date = 0
    end_date = 9000
    # choose a number of time steps
    d = pd.read_csv(f'{path}{file}.csv')
    d = d[:end_date]
    raw_seq = d[start_date:len(d) - n_steps * 2 - n_ahead][column_name].tolist()
    raw_seq, data_mean, data_std = normalize(array(raw_seq))

    # split into samples
    X, y = split_sequence(raw_seq, n_steps * 2)
    X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

    # demonstrate prediction
    x_input = array(d[len(d) - n_steps * 2 - n_ahead:len(d) - n_ahead][column_name]
                    .tolist())
    x_input, _, _ = normalize(x_input, data_mean, data_std, 1)

    true_value = d[len(d) - n_ahead:][column_name].tolist()

    return X, y, x_input, true_value, data_mean, data_std


n_seq = 2
n_steps = 24 * 5
n_features = 1

n_ahead = 24 * 2
n_days = 1

# define model
model = Sequential()
model.add(Bidirectional(ConvLSTM2D(
        filters=24,
        kernel_size=(1, 2),
        padding='same',
        activation='relu',
        input_shape=(n_seq, 1, n_steps, n_features)
)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(100))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=[RootMeanSquaredError(),
                       MeanSquaredError(),
                       MeanAbsoluteError(),
                       Accuracy()]
              )

column_name = 'value'

files = ['AEP_hourly', 'COMED_hourly', 'DAYTON_hourly', 'kms_2021_2023', 'DEOK_hourly',
         'DOM_hourly', 'DUQ_hourly', 'EKPC_hourly', 'FE_hourly', 'NI_hourly',
         'PJM_Load_hourly', 'PJME_hourly', 'PJMW_hourly']

# files = ['kms_2021_2023']
# files = ['openweathermap_komsomolsk_on_amur_2001_2022']
# files = ['DAYTON_hourly_0', 'DAYTON_hourly_1', 'DAYTON_hourly_2', 'DAYTON_hourly_3',
#          'DAYTON_hourly_4', 'DAYTON_hourly_5', 'DAYTON_hourly_6', 'DAYTON_hourly_7',
#          'DAYTON_hourly_8', 'DAYTON_hourly_9', 'DAYTON_hourly_10', 'DAYTON_hourly_11',
#          'DAYTON_hourly_12', 'DAYTON_hourly_13']

path = 'input\\full_datasets\\energy\\pjm_interconnection_llc\\'
print(files[0:4])
for _ in range(1, 2):
    for file in files[0:4]:
        X, y, x_input, true_value, data_mean, data_std = get_data(file, path, column_name,
                                                                  n_ahead, n_seq, n_steps,
                                                                  n_features)
        # fit model
        model.fit(X, y, epochs=10, batch_size=256,
                  # shuffle=True,
                  verbose=1)
        model.summary()

model.save('my_model.h5')

# model = load_model('my_model.h5', compile=False)

result = []

for _ in range(n_ahead):
    # Прогноз
    x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
    fc = model.predict(x_input, verbose=1)
    result.append(fc[0][0])

    # Создание новой входной матрицы для прогнозирования
    x_input = np.append(x_input, fc)
    # Исключение первой переменной
    x_input = np.delete(x_input, 0)
    # Изменение формы для следующей итерации
    x_input = np.reshape(x_input, (1, len(x_input), 1))

result = des_normalize(np.array(result), data_mean, data_std)

# Absolute error
a = []
for i in range(0, (n_ahead * n_days)):
    a.append(abs(true_value[i] - result[i]))

absolute_error = round(sum(a) / (n_ahead * n_days), 2)

# Relative errors
e = []
for i in range(0, (n_ahead * n_days)):
    e.append(abs(true_value[i] - result[i]) / true_value[i] * 100)

relative_error = round(abs(sum(e) / (n_ahead * n_days)))

# Составление прогнозов (график)
plt.figure(figsize=(15, 6))
plt.plot(true_value, c="blue", linewidth=3, label='Реальные значения',
         marker='o')

for x, y in zip(true_value[0:len(true_value):10],
                range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
    plt.text(y + .8, x + .2, str(x), color='blue')

plt.plot(result, c="green", linewidth=3, label='Прогнозируемые значения', marker='X')
for x, y in zip(result[0:len(result):10], range(0, (n_ahead * n_days))[0:n_ahead * n_days:10]):
    plt.text(y + .8, x + .2, str(x), color='green')

plt.title('Прогназирование')

plt.xticks(range(0, (n_ahead * n_days)))
plt.xlabel(f'Часы\n'
           f'Абсолютная ошибка: {absolute_error}, относительная ошибка: {relative_error}%')
plt.ylabel('Погода')

plt.grid()
plt.legend()

plt.show()
