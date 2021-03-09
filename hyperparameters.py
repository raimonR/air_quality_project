import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_set_x = np.load('dataset/train_set_x.npy')
train_set_y = np.load('dataset/train_set_y.npy')
dev_set_x = np.load('dataset/dev_set_x.npy')
dev_set_y = np.load('dataset/dev_set_y.npy')
# test_set_x = np.load('dataset/test_set_x.npy')
# test_set_y = np.load('dataset/test_set_y.npy')

inputs = keras.layers.Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
lstm_out = keras.layers.LSTM(units=1)(inputs)
outputs = keras.layers.Dense(units=24)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025), loss='mse')
model.summary()

epochs = [10, 50, 100, 200, 500, 1000, 2000, 4000]
t_loss = []
v_loss = []
for e in epochs:
    for i in range(5):
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), epochs=e, batch_size=4)
        t_loss.append(res.history['loss'])
        v_loss.append(res.history['val_loss'])

print("done")
