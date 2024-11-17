from CIFAR10 import CIFAR10
import tensorflow as tf
import numpy as np

data = CIFAR10(dataset_path="datasets/CIFAR10")

# LeNet-5 architecture
model = tf.keras.Sequential([
    # C1: Convolutional Layer
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same', activation="relu", input_shape=(32, 32, 3)),
    
    # S2: Average Pooling Layer
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    
    # C3: Convolutional Layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation="relu"),
    
    # S4: Average Pooling Layer
    tf.keras.layers.AveragePooling2D(pool_size=2, strides=2),
    
    # C5: Convolutional Layer (fully connected in LeNet-5)
    tf.keras.layers.Conv2D(filters=120, kernel_size=5, strides=1, padding='valid', activation="relu"),
    
    # Flatten layer to connect to fully connected layers
    tf.keras.layers.Flatten(),
    
    # F6: Fully Connected Layer
    tf.keras.layers.Dense(84, activation="relu"),
    
    # Output Layer for 10 classes
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
)

# Train the model
model.fit(np.array(data.X_train), np.array(data.Y_train), epochs=20, batch_size=5)

test_image = np.expand_dims(data.X_test[0], axis=0)
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Predicted class: {predicted_class}")
print(f"True class: {data.Y_test[0]}")