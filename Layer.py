import random

RANDOM_WEIGHT_RANGE = 0.1

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: int):
        super().__init__()
        self.weights = []
        self.biases = [random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE) for _ in range(filters)]
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    def forward(self, input):
        output = []
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
        if width != height:
            raise Exception("Input width must be the same as input height.")
        feature_map_size = (width + self.padding * 2 - self.kernel_size) / self.strides + 1
        if feature_map_size % 1 != 0 or feature_map_size < 1:
            raise Exception("Feature map size is negative or not an integer.")
        feature_map_size = int(feature_map_size)
        
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
                                _sum += filter_weight * input[input_k][input_j][input_k]
                    # Add bias
                    filter_bias = self.biases[filter_index]
                    feature_map[starting_i][starting_j] = _sum + filter_bias
            output.append(feature_map)
        self.output = output
        return output
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
        print(self.output)
        return output

            
c = Conv2D(6, 2, 1, 0)
step_1 = c.forward(
    [[[1.0 for _ in range(4)] for _ in range(4)] for _ in range(3)]
)
b = AveragePooling2D(2, 1)
step_2 = b.forward(step_1)


                    

        