from CIFAR10 import CIFAR10
from Layer import *

def to_onehot(size: int, y: int):
    return [1.0 if _ == y else 0.0 for _ in range(size)]
def main():
    data = CIFAR10('datasets/CIFAR10')
    # print(data.X_train[0], to_onehot(10, data.Y_train[0]))
    Y_train_hotcodded = [to_onehot(10, y) for y in data.Y_train]
    # LeNet
    # model = Sequential([
    #       Conv2D(filters=6, kernel_size=5, strides=1, padding='same', activation='relu'),
    #       AveragePooling2D(pool_size=2, strides=2),
    #       Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'),
    #       AveragePooling2D(pool_size=2, strides=2),
    #       Conv2D(filters=120, kernel_size=5, strides=1, padding='valid', activation='relu'),
    #       Flatten(),
    #       Dense(84, 'relu'),
    #       Dense(10, 'softmax')
    #     ])
    # Shallow CNN
    # model = Sequential([
    #     Conv2D(filters=4, kernel_size=3, strides=1, padding='valid', activation='relu'),
    #     AveragePooling2D(pool_size=2, strides=2),
    #     Flatten(),
    #     Dense(32, 'relu'),
    #     Dense(10, 'softmax')
    # ])
    model = Sequential([
        # First Convolutional Block
        Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        AveragePooling2D(pool_size=2, strides=2),

        # Second Convolutional Block
        Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        AveragePooling2D(pool_size=2, strides=2),

        # Fully Connected Layers
        Flatten(),
        Dense(256, 'relu'),  # Increased number of neurons for more capacity
        Dense(128, 'relu'),
        Dense(10, 'softmax')  # Output layer for 10 classes
    ])
    input_shape = [3, 32, 32]
    model.compile(optimizer='sgd', loss='cce', learning_rate=0.1, input_shape=input_shape)
    # overfitting data for testing
    model.fit(data.X_train[:10], Y_train_hotcodded[:10], epochs=200)
if __name__ == '__main__':
    main()