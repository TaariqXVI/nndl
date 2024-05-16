import numpy as np
import tensorflow as tf

# Generate some synthetic data
np.random.seed(42)
X_train = np.random.rand(100, 1)  # Input features
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)  # Linear relation with some noise

# Build the regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,), activation='linear')
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8)

# Generate some test data
X_test = np.array([[0.2], [0.5], [0.8]])

# Make predictions
predictions = model.predict(X_test)

# Display the predictions
print("Predictions:")
print(predictions)
