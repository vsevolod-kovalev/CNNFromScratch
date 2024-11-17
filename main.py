from CIFAR10 import CIFAR10

def main():
    data = CIFAR10(dataset_path = "datasets/CIFAR10")
    print(data.X_train[0][0][0], data.Y_train[0])

if __name__ == '__main__':
    main()