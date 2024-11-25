from CIFAR10 import CIFAR10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

def to_onehot(size: int, y: int):
    return [1.0 if _ == y else 0.0 for _ in range(size)]

learning_rate = 0.01
optimizer = SGD(learning_rate=learning_rate)
data = CIFAR10(dataset_path="datasets/CIFAR10")
Y_train_hotcodded = [to_onehot(10, y) for y in data.Y_train]
model = Sequential([
    Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)),
    AveragePooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
# Compile the model
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
)

# Train the model
model.fit(np.array(data.X_train[:300]), np.array(Y_train_hotcodded[:300]), epochs=30, batch_size=1)

test_image = np.expand_dims(data.X_test[0], axis=0)
prediction = model.predict(test_image)
predicted_class = np.argmax(prediction, axis=1)[0]

print(f"Predicted class: {predicted_class}")
print(f"True class: {data.Y_test[0]}")