import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras as kr

temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")

sns.scatterplot(x='Celsius', y='Fahrenheit', data=temperature_df)


x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']


model = kr.Sequential()
model.add(kr.layers.Dense(units=1, input_shape=[1]))


adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.9)

model.summary()

model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs = 100)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])

plt.title('Grafico de prueba')
plt.xlabel('Epoch')
plt.ylabel('perdida')
plt.legend('perdida')


Temp_C = 0
Temp_F = model.predict([Temp_C])
print("Temperatura de Prediccion: " + str(Temp_F))


Temp_F = 9/5 * Temp_C + 32
print("Temperatura de Ecuacion: " + str(Temp_F))