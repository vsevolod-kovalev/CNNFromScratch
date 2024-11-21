import random

RANDOM_WEIGHT_RANGE = 0.1

# assuming l has even row and column distibution
def shape(l):
    shape = []
    while True:
        if isinstance(l, list):
            shape.append(len(l))
            l = l[0]
        else:
            break
    return shape

class Layer:
    def __init__(self):
        self.input = None
        self.preactivation = None
        self.output = None

class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, activation_function: str):
        super().__init__()
        self.weights = []
        self.biases = [random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE) for _ in range(filters)]
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        if not (padding == 'same' or padding == 'valid'):
            raise Exception("Unknown padding format. Conv2D padding can be \'valid\' or \'same\'.")
        self.padding = padding
        self.activation_function = activation_function

    # Activation functions
    @staticmethod
    def reLU(layer, derivative: bool = False):
        depth, height, width = len(layer), len(layer[0]), len(layer[0][0])
        output  = [
            [[0.0 for _ in range(width)]
                  for __ in range(height)]
                  for ___ in range(depth)
        ]
        relu = lambda z: (1.0 if z > 0 else 0.0) if derivative else (max(0, z))
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    z = layer[i][j][k]
                    output[i][j][k] = relu(z)
        return output
    @staticmethod
    def activate(layer, activation: str):
        match activation.lower():
            case 'relu':
                return Conv2D.reLU(layer)
            case _:
                raise Exception("Unknown activation function")
    @staticmethod
    def add_padding(input, padding):
        depth, height, width = len(input), len(input[0]), len(input[0][0])
        padded_input = [
            [[0.0 for _ in range(width + 2 * padding)]
                  for __ in range(height + 2 * padding)]
                  for ___ in range(depth)
        ]
        for i in range(depth):
            for j in range(height):
                for k in range(width):
                    padded_input[i][j + padding][k + padding] = input[i][j][k]
        return padded_input
    def forward(self, input):
        depth, height, width = len(input), len(input[0]), len(input[0][0])
        if not self.weights:
            self.weights = [
                [
                    [
                        [random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE)
                        for _ in range(depth)]
                        for _ in range(self.kernel_size)
                    ]
                    for _ in range(self.kernel_size)
                ]
                for _ in range(self.num_filters)
            ]
        padding = 0
        if self.padding == 'same':
            # calculate the neccesary padding to maintain the input size:
            padding = (self.strides * width - self.strides - width + self.kernel_size) / 2
        if padding % 1 != 0 or padding < 0:
            raise Exception("Padding must be a non-negative integer.")
        padding = int(padding)
        Z = []
        if width != height:
            raise Exception("Input width must be the same as input height.")
        feature_map_size = (width + padding * 2 - self.kernel_size) / self.strides + 1
        if feature_map_size % 1 != 0 or feature_map_size < 1:
            raise Exception("Feature map size must be a positive integer.")
        feature_map_size = int(feature_map_size)
        if padding:
            input = Conv2D.add_padding(input, padding)
  
        for filter_index in range(self.num_filters):
            feature_map = [
                [0.0 for _ in range(feature_map_size)] for __ in range(feature_map_size)
            ]
            # Convolution
            for starting_i in range(feature_map_size):
                for starting_j in range(feature_map_size):
                    _sum = 0.0
                    # Apply the kernel
                    for weight_i in range(self.kernel_size):
                        for weight_j in range(self.kernel_size):
                            for input_k in range(depth):
                                input_i = starting_i * self.strides + weight_i
                                input_j = starting_j * self.strides + weight_j
                                filter_weight = self.weights[filter_index][weight_i][weight_j][input_k]
                                _sum += filter_weight * input[input_k][input_i][input_j]
                    # Add bias
                    filter_bias = self.biases[filter_index]
                    feature_map[starting_i][starting_j] = _sum + filter_bias
            Z.append(feature_map)
        self.preactivation = Z
        self.output = Conv2D.activate(Z, self.activation_function)
        return self.output
class AveragePooling2D(Layer):
    def __init__(self, pool_size: int, strides: int):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
    def forward(self, input):
        output = []
        depth, height, width = len(input), len(input[0]), len(input[0][0])
        if width != height:
            raise Exception("Input width must be the same as input height.")
        feature_map_size = (width - self.pool_size) / self.strides + 1
        if feature_map_size % 1 != 0 or feature_map_size < 1:
            raise Exception("Feature map size is negative or not an integer.")
        feature_map_size = int(feature_map_size)
        pool_area = self.pool_size * self.pool_size
        for channel_index in range(depth):
            feature_map = [
                [0.0 for _ in range(feature_map_size)] for __ in range(feature_map_size)
            ]
            for output_i in range(feature_map_size):
                for output_j in range(feature_map_size):
                    _sum = 0.0
                    # Calculate the starting indices of the pooling window
                    start_i = output_i * self.strides
                    start_j = output_j * self.strides
                    # Sum over the pooling window
                    for i in range(start_i, start_i + self.pool_size):
                        for j in range(start_j, start_j + self.pool_size):
                            _sum += input[channel_index][i][j]
                    feature_map[output_i][output_j] = _sum / pool_area
            output.append(feature_map)
        self.output = output
        return output

c = Conv2D(filters = 6, kernel_size = 3, strides = 1, padding = 'same', activation_function = 'relu')
step_1 = c.forward(
    [[[1.0 for _ in range(4)] for _ in range(4)] for _ in range(3)]
)
b = AveragePooling2D(pool_size = 2, strides = 2)
step_2 = b.forward(step_1)
print(step_2, shape(step_2))


                    

        