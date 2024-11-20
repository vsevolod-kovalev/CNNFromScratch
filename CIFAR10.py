import pickle
import os

class CIFAR10:
    def load_cifar10_batch(self, file_path):
        with open(file_path, 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            images = batch[b'data']
            labels = batch[b'labels']
            
            # image input format:       channel  x   width   x   height
            images = [images[i].reshape(3, 32, 32) / 255.0 for i in range(len(images))]
        
        return images, labels

    def load_cifar10_dataset(self, dataset_path):
        train_images = []
        train_labels = []
        # Load the first batch
        for i in range(1, 2):
            file_path = os.path.join(dataset_path, f'data_batch_{i}')
            images, labels = self.load_cifar10_batch(file_path)
            train_images.extend(images)
            train_labels.extend(labels)
        # Load the test batch
        test_images, test_labels = self.load_cifar10_batch(os.path.join(dataset_path, 'test_batch'))
        
        return (train_images, train_labels), (test_images, test_labels)

    def __init__(self, dataset_path):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = self.load_cifar10_dataset(dataset_path)
