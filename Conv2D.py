from Layer import *

class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: str, activation: str):
        super().__init__()
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        if padding.lower() not in ('same', 'valid'):
            raise Exception("Unknown padding format. Conv2D padding can be 'valid' or 'same'.")
        self.padding = padding.lower()
        self.padding_value = 0
        self.activation_function = activation
        # Variables to be initialized in compile()
        self.input_depth = None
        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None
        self.preactivation = None
        self.forward_output = None
        self.backward_output = None
        self.dA_dZ = None
    @staticmethod
    def add_padding(input, padding):
        depth, height, width = len(input), len(input[0]), len(input[0][0])
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        padded_input = [
            [[0.0 for _ in range(padded_width)]
                for __ in range(padded_height)]
            for ___ in range(depth)
        ]
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    padded_input[d][h + padding][w + padding] = input[d][h][w]
        return padded_input

    def reLU(self, Z, A, derivative: bool = False):
        depth, height, width = len(Z), len(Z[0]), len(Z[0][0])
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    z = Z[d][h][w]
                    if derivative:
                        A[d][h][w] = 1.0 if z > 0 else 0.0
                    else:
                        A[d][h][w] = max(0.0, z)

    def compile(self, input_shape):
        self.input_depth, self.input_height, self.input_width = input_shape
        # Initialize weights and biases
        self.biases = [random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE) for _ in range(self.num_filters)]
        self.weights = [
            [
                [
                    [
                        random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE) for _ in range(self.kernel_size)
                    ]
                    for _ in range(self.kernel_size)
                ]
                for _ in range(self.input_depth)
            ]
            for _ in range(self.num_filters)
        ]
        # Calculate padding
        if self.padding == 'same':
            self.padding_value = (self.kernel_size - 1) // 2
        elif self.padding == 'valid':
            self.padding_value = 0

        # Calculate output dimensions
        padded_height = self.input_height + 2 * self.padding_value
        padded_width = self.input_width + 2 * self.padding_value
        self.output_height = ((padded_height - self.kernel_size) // self.strides) + 1
        self.output_width = ((padded_width - self.kernel_size) // self.strides) + 1
        if self.output_height <= 0 or self.output_width <= 0:
            raise Exception("Invalid output dimensions. Check your kernel size, padding, and strides.")

        # Preallocate arrays
        self.preactivation = [
            [[0.0 for _ in range(self.output_width)] for __ in range(self.output_height)]
            for ___ in range(self.num_filters)
        ]
        self.forward_output = [
            [[0.0 for _ in range(self.output_width)] for __ in range(self.output_height)]
            for ___ in range(self.num_filters)
        ]
        self.backward_output = [
            [[0.0 for _ in range(self.input_width)] for __ in range(self.input_height)]
            for ___ in range(self.input_depth)
        ]
        self.dA_dZ = [
            [[0.0 for _ in range(self.output_width)] for __ in range(self.output_height)]
            for ___ in range(self.num_filters)
        ]
        # Initialize deltas
        self.weight_deltas = [
            [
                [
                    [0.0 for _ in range(self.kernel_size)] for __ in range(self.kernel_size)
                ]
                for ___ in range(self.input_depth)
            ]
            for ____ in range(self.num_filters)
        ]
        self.bias_deltas = [0.0 for _ in range(self.num_filters)]
        return (self.num_filters, self.output_height, self.output_width)

    def backward(self, gradient):
        self.backward_output = [
            [[0.0 for _ in range(self.input_width)] for __ in range(self.input_height)]
            for ___ in range(self.input_depth)
        ]
        self.weight_deltas = [
                    [
                        [
                            [0.0 for _ in range(self.kernel_size)] for __ in range(self.kernel_size)
                        ]
                        for ___ in range(self.input_depth)
                    ]
                    for ____ in range(self.num_filters)
        ]
        self.bias_deltas = [0.0 for _ in range(self.num_filters)]

        # Compute derivative of activation function
        self.activate(self.preactivation, self.dA_dZ, self.activation_function, derivative=True)

        # Backpropagation
        padded_input = self.input
        if self.padding_value > 0:
            padded_input = self.add_padding(self.input, self.padding_value)
        for f in range(self.num_filters):
            for h_out in range(self.output_height):
                for w_out in range(self.output_width):
                    dL_dz = gradient[f][h_out][w_out] * self.dA_dZ[f][h_out][w_out]
                    self.bias_deltas[f] += dL_dz
                    for d in range(self.input_depth):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                h_in = h_out * self.strides + kh
                                w_in = w_out * self.strides + kw
                                input_val = padded_input[d][h_in][w_in]
                                self.weight_deltas[f][d][kh][kw] += dL_dz * input_val
                                # Calculate gradient to pass to previous layer
                                weight = self.weights[f][d][kh][kw]
                                if 0 <= h_in - self.padding_value < self.input_height and 0 <= w_in - self.padding_value < self.input_width:
                                    self.backward_output[d][h_in - self.padding_value][w_in - self.padding_value] += dL_dz * weight
        return self.backward_output

    def forward(self, input):
        self.input = input
        padded_input = input
        if self.padding_value > 0:
            padded_input = self.add_padding(input, self.padding_value)
        # Convolution
        for f in range(self.num_filters):
            for h_out in range(self.output_height):
                for w_out in range(self.output_width):
                    sum = 0.0
                    for d in range(self.input_depth):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                h_in = h_out * self.strides + kh
                                w_in = w_out * self.strides + kw
                                input_val = padded_input[d][h_in][w_in]
                                weight = self.weights[f][d][kh][kw]
                                sum += input_val * weight
                    self.preactivation[f][h_out][w_out] = sum + self.biases[f]
        # Apply activation function
        self.activate(self.preactivation, self.forward_output, self.activation_function)
        return self.forward_output