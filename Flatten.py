from Layer import *

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_depth = None
        self.input_height = None
        self.input_width = None
        self.forward_output = None
        self.backward_output = None
        self.flattened_index = 0
    def compile(self, input_shape):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.forward_output = [0.0 for _ in range(self.input_depth * self.input_height * self.input_width)]
        self.backward_output = [
            [[0.0 for _ in range(self.input_width)]
                  for __ in range(self.input_height)]
                  for ___ in range(self.input_depth)
        ]
        return Layer.shape(self.forward_output)
    def backward(self, gradient):
        # 1D List   ->     3D Tensor
        self.flattened_index = 0
        for depth_index in range(self.input_depth):
            for height_index in range(self.input_height):
                for width_index in range(self.input_width):
                    self.backward_output[depth_index][height_index][width_index] = gradient[self.flattened_index]
                    self.flattened_index += 1
        return self.backward_output
    def forward(self, input):
        # 3D Tensor ->     1D List
        self.flattened_index = 0
        for depth_index in range(self.input_depth):
            for height_index in range(self.input_height):
                for width_index in range(self.input_width):
                    self.forward_output[self.flattened_index] = input[depth_index][height_index][width_index] 
                    self.flattened_index += 1
        return self.forward_output