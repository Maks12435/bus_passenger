import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def build_and_train_nn(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    y_pred = model.predict(X_test).flatten()

    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Neural Network Predictions')
    plt.xlabel('Actual Total Travel Time')
    plt.ylabel('Predicted Total Travel Time')
    plt.xlim(y_test.min(), y_test.max())
    plt.ylim(y_test.min(), y_test.max())
    plt.grid()
    plt.show()

    return model
