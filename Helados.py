# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:29:59 2023

@author: Jarez
"""

import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as kr

sales_df = pd.read_csv("datos_de_ventas.csv")

sns.scatterplot(x='Temperature', y='Revenue', data=sales_df)

x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

model = kr.Sequential()
model.add(kr.layers.Dense(units=1, input_shape = [1]))

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.summary()

model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs = 1000)

keys = epochs_hist.history.keys()


plt.plot(epochs_hist.history['loss'])

plt.title('Grafica de helados')
plt.xlabel('Epoch')
plt.ylabel('perdida')
plt.legend('perdida')

weights = model.get_weights()


Temp = 30
Revenue =  model.predict([Temp])
print(f'la prediccion de  la ganacias es de : {Revenue}')

plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color = 'red')
plt.ylabel('Ganancias [dolares]')
plt.xlabel('Temperatura [gCelsius]')
plt.title('Ganancia generada VS. Temperatura @helados S.A')