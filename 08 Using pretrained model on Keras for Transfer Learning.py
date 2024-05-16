from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess data by normalizing pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Load pre-trained VGG16 model (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze pre-trained model weights
for layer in base_model.layers:
  layer.trainable = False

# Define custom top layers for transfer learning
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))  # 10 output units for 10 CIFAR-10 classes

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
model.compile(optimizer=optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model architecture summary
model.summary()

# Create data generator with validation split for image augmentation
datagen = ImageDataGenerator(validation_split=0.2)
batch_size = 32

# Use data generators for training and validation with image augmentation
train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
validation_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')

# Train the model with early stopping based on validation performance (optional)
epochs = 10  # Adjust epochs as needed
history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epochs,
                    validation_data=validation_generator, validation_steps=len(validation_generator))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
