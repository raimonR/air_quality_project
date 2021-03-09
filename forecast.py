import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# TODO: LOAD HYPERPARAMETERS FROM PARAMETERS.JSON TO SET BATCH SIZE AND LEARNING RATE

train_set_x = np.load('dataset/train_set_x.npy')
train_set_y = np.load('dataset/train_set_y.npy')
# dev_set_x = np.load('dataset/dev_set_x.npy')
# dev_set_y = np.load('dataset/dev_set_y.npy')
test_set_x = np.load('dataset/test_set_x.npy')
test_set_y = np.load('dataset/test_set_y.npy')

inputs = keras.layers.Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
lstm_out = keras.layers.LSTM(units=128, return_sequences=True)(inputs)
lstm_out = keras.layers.LSTM(units=64)(lstm_out)
outputs = keras.layers.Dense(units=24)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
model.summary()

res = model.fit(x=train_set_x, y=train_set_y, epochs=50, batch_size=256)

forecast = model.predict(x=test_set_x)

fig, axes = plt.subplots(nrows=6, ncols=6, sharex=True, sharey=True)
plt.title('orange=forecast, blue=true')
for i, ax in enumerate(axes.flatten()):
    ax.plot(range(24), test_set_y[i], color='tab:blue')
    ax.plot(range(24), forecast[i], color='tab:orange')

plt.show()


print("done")
