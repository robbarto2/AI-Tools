import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 2 * X ** 2 + 1 + np.random.normal(0, 1, 100)

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(10)

# Build the model
model = Sequential([
    Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Create a TensorBoard callback
log_dir = "logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(dataset, epochs=50, callbacks=[tensorboard_callback])

# Load TensorBoard in your browser using the following command:
# tensorboard --logdir=logs/
