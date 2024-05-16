import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer  # Updated import for TensorFlow 2.x
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Updated import for TensorFlow 2.x
from tensorflow.keras.utils import to_categorical  # Updated import for TensorFlow 2.x
from tensorflow.keras.models import Sequential  # Updated import for TensorFlow 2.x
from tensorflow.keras.layers import SimpleRNN, Dense, Activation  # Updated imports for TensorFlow 2.x
from tensorflow.keras import optimizers  # Updated import for TensorFlow 2.x
from tensorflow.keras.metrics import categorical_accuracy  # Updated import for TensorFlow 2.x
import numpy as np

# Load Yelp dataset
df = pd.read_csv('yelp.csv')

# Separate features and target labels
x = df['sentence']
y = df['label']

# Split data into training and testing sets (80%/20% split)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Text pre-processing: Tokenization
tokenizer = Tokenizer(num_words=1000, lower=True)
tokenizer.fit_on_texts(X_train)

# Convert sentences to sequences of integers (word indexes)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length for consistent model input
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# One-hot encode labels for categorical crossentropy loss
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Reshape input data for RNN (samples, sequence length, features)
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Define a simple RNN model
def vanilla_rnn():
  model = Sequential()
  model.add(SimpleRNN(50, input_shape=(maxlen, 1), return_sequences=False))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  model.summary()
  adam = optimizers.Adam(learning_rate=0.001)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
  return model

# Train the model using KerasClassifier wrapper
from tensorflow.keras.wrappers import KerasClassifier  # Updated import for TensorFlow 2.x
model = KerasClassifier(build_fn=vanilla_rnn, epochs=5, batch_size=50)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = np.mean(categorical_accuracy(y_test, y_pred))
print("Accuracy:", accuracy)
