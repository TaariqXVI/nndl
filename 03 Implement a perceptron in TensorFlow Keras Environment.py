import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some synthetic data for a binary classification task
np.random.seed(42)
X = np.random.rand(100, 2)  # Input features (2 features for simplicity)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary label based on a simple condition

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a perceptron model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(2,), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy')  # Include accuracy metric if needed

# Train the perceptron
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Make predictions on the test set
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Evaluate accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy on the test set:", accuracy)
