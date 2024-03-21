# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:13:41 2023

@author: Jarez
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importando Datos
house_df = pd.read_csv("precios_hogares.csv")

# Convertir la columna 'date' a formato numérico (si es una fecha en el formato '20141013T000000')
house_df['date'] = pd.to_numeric(house_df['date'], errors='coerce')

# VISUALIZACION
sns.scatterplot(x='sqft_living', y='price', data=house_df)

# Correlación sin la columna 'date'
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(house_df.drop(['date'], axis=1).corr(), annot=True)


selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

X = house_df[selected_features]
y = house_df['price']


from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

y = y.values.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

model = tf.keras.models.Sequential()

# Add layers to the model
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7,)))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Display the model summary
model.summary()

model.compile(optimizer='Adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del Modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

X_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])

scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)

y_predict_1 = model.predict(X_test_scaled_1)

y_predict_1 = y_predict_1.reshape(1, -1)

y_predict_1 = scaler_1.inverse_transform(y_predict_1.reshape(1, -1))

y_predict_1 = y_predict_1.reshape(-1, 1)

print(y_predict_1)



