
import tensorflow as tf 
import pandas as pd 
import numpy as np 

from tensorflow import keras
from sklearn.model_selection import train_test_split

# consider X as house size in square feet

X = np.array([
    [1000],
    [1200],
    [1500],
    [1800],
    [2000]
])

#consider Y as price in lacks 

y = np.array([
    50,
    60,
    75,
    90,
    100
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2, # 20% of the data is used for testing and 80% is used for traing
    random_state = 42
)

#building the architecture of the model
model = keras.Sequential([
    keras.Input(shape = (1,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

#compliling the model
model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mae']
)

#training a model

history = model.fit(
    X_train,
    y_train,
    epochs = 100,
    batch_size = 2
)

#evavulate the model

loss, mae = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("MAE:", mae)

house_size = float(input("Enter the size of the house: "))

input_data = np.array([[house_size]])

prediction = model.predict(input_data)

print("Predicted House Price:", prediction[0][0])