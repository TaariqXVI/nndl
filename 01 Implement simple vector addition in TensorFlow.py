import tensorflow as tf

# Define two vectors
vector1 = tf.constant([1.0, 2.0, 3.0])
vector2 = tf.constant([4.0, 5.0, 6.0])

# Perform vector addition
result_vector = tf.add(vector1, vector2)

# Print the results
print("Vector 1:", vector1.numpy())
print("Vector 2:", vector2.numpy())
print("Resultant Vector:", result_vector.numpy())
