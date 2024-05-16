import pandas as pd
import numpy as np
from faker import Faker
import random
import datetime

fake = Faker()

# Generate sample users
num_users = 100
users = [fake.name() for _ in range(num_users)]

# Generate sample items
num_items = 50
items = [fake.word() for _ in range(num_items)]

# Generate sample sales data
num_transactions = 500
data = {
    'user': [random.choice(users) for _ in range(num_transactions)],
    'item': [random.choice(items) for _ in range(num_transactions)],
    'purchase': [random.choice([0, 1]) for _ in range(num_transactions)],
    'timestamp': [fake.date_time_between(start_date="-1y", end_date="now") for _ in range(num_transactions)]
}

sales_data = pd.DataFrame(data)

# Save the generated data to a CSV file
sales_data.to_csv('sample_sales_data.csv', index=False)

# Display the first few rows of the generated data
print(sales_data.head())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Load the sales data
data = pd.read_csv('/content/sample_sales_data.csv')

# Preprocess data
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user'])
data['item_id'] = item_encoder.fit_transform(data['item'])

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the model
def create_model(num_users, num_items, embedding_size=50):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, input_length=1)(item_input)

    user_flatten = Flatten()(user_embedding)
    item_flatten = Flatten()(item_embedding)

    concat = Concatenate()([user_flatten, item_flatten])
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
num_users = len(data['user_id'].unique())
num_items = len(data['item_id'].unique())

model = create_model(num_users, num_items)
model.summary()

train_user = train_data['user_id'].values
train_item = train_data['item_id'].values
train_labels = train_data['purchase'].values

model.fit([train_user, train_item], train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
test_user = test_data['user_id'].values
test_item = test_data['item_id'].values
test_labels = test_data['purchase'].values

accuracy = model.evaluate([test_user, test_item], test_labels)
print(f'Test Accuracy: {accuracy[1]*100:.2f}%')
