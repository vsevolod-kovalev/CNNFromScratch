from Layer import *
class AveragePooling2D(Layer):
    def __init__(self, pool_size: int, strides: int):
        super().__init__()
        self.pool_size = pool_size
        self.pool_area = None
        self.output_height = None
        self.output_width = None
        self.input_depth = None
        self.input_height = None
        self.input_width = None
        self.strides = strides
        self.forward_output = None
        self.backward_output = None

    def compile(self, input_shape):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.output_height = (self.input_height - self.pool_size) // self.strides + 1
        self.output_width = (self.input_width - self.pool_size) // self.strides + 1
        if self.output_height <= 0 or self.output_width <= 0:
            raise Exception("Invalid output dimensions for AveragePooling2D. Check your pool size and strides.")
        self.pool_area = self.pool_size * self.pool_size
        self.forward_output = [
            [
                [0.0 for _ in range(self.output_width)] for __ in range(self.output_height)
            ] for ___ in range(self.input_depth)
        ]
        self.backward_output = [
            [
                [0.0 for _ in range(self.input_width)] for __ in range(self.input_height)
            ] for ___ in range(self.input_depth)
        ]
        return (self.input_depth, self.output_height, self.output_width)

    def backward(self, gradient):
        self.backward_output = [
            [
                [0.0 for _ in range(self.input_width)] for __ in range(self.input_height)
            ] for ___ in range(self.input_depth)
        ]
        # Averaged 3D tensor -> Initial 3D tensor
        for depth_index in range(self.input_depth):
            for averaged_map_i in range(self.output_height):
                for averaged_map_j in range(self.output_width):
                    dL_dAvg = gradient[depth_index][averaged_map_i][averaged_map_j]
                    start_i = averaged_map_i * self.strides
                    start_j = averaged_map_j * self.strides
                    for input_i in range(start_i, min(start_i + self.pool_size, self.input_height)):
                        for input_j in range(start_j, min(start_j + self.pool_size, self.input_width)):
                            self.backward_output[depth_index][input_i][input_j] += dL_dAvg / self.pool_area
        return self.backward_output

    def forward(self, input):
        self.input = input
        for depth_index in range(self.input_depth):
            for output_i in range(self.output_height):
                for output_j in range(self.output_width):
                    _sum = 0.0
                    start_i = output_i * self.strides
                    start_j = output_j * self.strides
                    for i in range(start_i, min(start_i + self.pool_size, self.input_height)):
                        for j in range(start_j, min(start_j + self.pool_size, self.input_width)):
                            _sum += input[depth_index][i][j]
                    self.forward_output[depth_index][output_i][output_j] = _sum / self.pool_area
        return self.forward_output
