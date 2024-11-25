from CIFAR10 import CIFAR10
from CIFAR10_display_utils import *
from Sequential import *
from Conv2D import *
from AveragePooling2D import *
from Flatten import *
from Dense import *

def to_onehot(size: int, y: int):
    return [1.0 if _ == y else 0.0 for _ in range(size)]

def main():
    data = CIFAR10('datasets/CIFAR10')
    model = Sequential.load_model_from_pickle('trained_model.pkl')
    for i in range(0, 10):
        output_hotcodded = model.predict(data.X_train[i])
        confidence = max(output_hotcodded)
        predicted_class = output_hotcodded.index(confidence)
        display_image_with_prediction(
            image=data.X_train[i],
            true_label=data.Y_train[i],
            predicted_class=predicted_class,
            confidence=confidence,
            index=i,
            show_greyscale=False,
            show_ascii=False
        )
        '''
        Uncomment to train:
        Y_train_hotcodded = [to_onehot(10, y) for y in data.Y_train]
        model = Sequential([
                Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'),
                AveragePooling2D(pool_size=2, strides=2),
                Flatten(),
                Dense(64, 'relu'),
                Dense(10, 'softmax') 
            ])

        input_shape = [3, 32, 32]
        model.compile(optimizer='sgd', loss='cce', learning_rate=0.01, input_shape=input_shape)
        model.fit(data.X_train[:300], Y_train_hotcodded[:300], epochs=30)
        '''
if __name__ == '__main__':
    main()
