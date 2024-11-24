import random
import math
import sys
import time
from typing import List

RANDOM_WEIGHT_RANGE = 0.1


def compute_stats(tensor):
    total = 0.0
    count = 0
    min_val = float('inf')
    max_val = float('-inf')
    stack = [tensor]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            val = current
            total += val
            count += 1
            min_val = min(min_val, val)
            max_val = max(max_val, val)
    mean_val = total / count if count > 0 else 0.0
    return min_val, max_val, mean_val

class Layer:
    def __init__(self):
        self.input = None
        self.preactivation = None
        self.output = None
        self.weight_deltas = None
        self.bias_deltas = None
        self.weights = None
        self.biases = None
    def activate(self, Z, A, activation: str, derivative: bool = False):
        activation = activation.lower()
        if activation == 'relu':
            self.reLU(Z, A, derivative=derivative)
        elif activation == 'softmax':
            if derivative:
                raise Exception("Derivative of softmax should not be computed directly.")
            else:
                self.softmax(Z, A)
        else:
            raise Exception(f"Unknown activation function '{activation}'")
    # assuming l has even row and column distibution
    @staticmethod
    def shape(_list):
        shape = []
        while True:
            if isinstance(_list, list):
                shape.append(len(_list))
                _list = _list[0]
            else:
                break
        return shape

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
class Dense(Layer):
    def __init__(self, num_neurons: int, activation: str):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation_function = activation
        self.forward_output = None
        self.backward_output = None
        self.input = None
        self.preactivation = None
    def reLU(self, Z, A, derivative: bool = False):
        relu = lambda z: (1.0 if z > 0 else 0.0) if derivative else (max(0, z))
        for i, z in enumerate(Z):
            # modifying provided activation array in-place
            A[i] = relu(z)
    def softmax(self, Z, A):
        # Shift values for numerical stability
        z_max = max(Z)
        Z_exp = [math.exp(z - z_max) for z in Z]
        z_sum = sum(Z_exp)
        for i, z in enumerate(Z_exp):
            A[i] = z / z_sum
    def compile(self, input_shape):
        self.input_num_neurons = input_shape[0]
        self.preactivation = [0.0 for _ in range(self.num_neurons)]
        self.dA_dZ = [0.0 for _ in range(self.num_neurons)]
        self.forward_output = [0.0 for _ in range(self.num_neurons)]
        self.backward_output = [0.0 for _ in range(self.input_num_neurons)]
        self.biases = [random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE) for _ in range(self.num_neurons)]
        self.weights = [[random.uniform(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE)
                        for _ in range(self.input_num_neurons)]
                        for __ in range(self.num_neurons)]
        self.weight_deltas = [[0.0 for _ in range(self.input_num_neurons)] for _ in range(self.num_neurons)]
        self.bias_deltas = [0.0 for _ in range(self.num_neurons)]
        return (self.num_neurons,)

    def backward(self, gradient):
        self.backward_output = [0.0 for _ in range(self.input_num_neurons)]
        dL_dz = None
        if self.activation_function.lower() == 'softmax':
            # For softmax with cross-entropy loss, the gradient is already computed correctly
            dL_dz = gradient
        else:
            self.activate(self.preactivation, self.dA_dZ, self.activation_function, derivative=True)
            dL_dz = [gradient[i] * self.dA_dZ[i] for i in range(self.num_neurons)]
        # Compute weight and bias deltas
        for neuron_index in range(self.num_neurons):
            self.bias_deltas[neuron_index] = dL_dz[neuron_index]
            for weight_index in range(self.input_num_neurons):
                dz_dw = self.input[weight_index]
                self.weight_deltas[neuron_index][weight_index] = dL_dz[neuron_index] * dz_dw
                self.backward_output[weight_index] += dL_dz[neuron_index] * self.weights[neuron_index][weight_index]
        return self.backward_output

    def forward(self, input):
        self.input = input
        # compute neurons pre-activations:
        for neuron_index in range(self.num_neurons):
            # number of weights in each neuron is equal
            # to the size of the flattened input:
            self.preactivation[neuron_index] = sum([
                input[input_index] * self.weights[neuron_index][input_index]
                for input_index in range(self.input_num_neurons)
            ]) + self.biases[neuron_index]
        self.activate(self.preactivation, self.forward_output, self.activation_function)
        return self.forward_output
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


DEFAULT_LEARNING_RATE = 0.01
LOG_EPSILON = 1e-15

class Sequential:
    def __init__(self, layers: List['Layer']):
        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.learning_rate = DEFAULT_LEARNING_RATE

    def compile(self, optimizer: str, loss: str, learning_rate: float = DEFAULT_LEARNING_RATE, input_shape=None):
        self.learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.loss = loss
        current_input_shape = input_shape
        for layer in self.layers:
            if hasattr(layer, 'compile'):
                if current_input_shape is None:
                    raise Exception("Input shape must be specified for the first layer.")
                current_input_shape = layer.compile(current_input_shape)
            else:
                raise Exception("Layer does not have a compile method.")

    @staticmethod
    def apply_deltas(weights, biases, weight_deltas, bias_deltas, learning_rate, weight_shape):
        # 2D Tensor for Dense layers.
        # 4D for Conv2D layers.
        if len(weight_shape) == 2:
            for neuron_index in range(weight_shape[0]):
                biases[neuron_index] -= learning_rate * bias_deltas[neuron_index]
                for weight_index in range(weight_shape[1]):
                    weights[neuron_index][weight_index] -= learning_rate * weight_deltas[neuron_index][weight_index]
        elif len(weight_shape) == 4:
            for filter_index in range(weight_shape[0]):
                biases[filter_index] -= learning_rate * bias_deltas[filter_index]
                for depth_index in range(weight_shape[1]):
                    for height_index in range(weight_shape[2]):
                        for width_index in range(weight_shape[3]):
                            weights[filter_index][depth_index][height_index][width_index] -= learning_rate * weight_deltas[filter_index][depth_index][height_index][width_index]

    def CCE_loss(self, Y, Y_hat) -> float:
        loss = 0.0
        for y, y_hat in zip(Y, Y_hat):
            # Prevent log(0) by adding a small epsilon
            y_hat_safe = y_hat + LOG_EPSILON
            loss += y * math.log(y_hat_safe)
        return -loss
    def fit(self, X, Y, epochs: int = 1):
        i = 0
        optimizer = self.optimizer.lower()
        if optimizer in ('sgd', 'stochastic_gradient_descent'):
            for epoch_index in range(epochs):
                print(f"\nEpoch {epoch_index + 1}/{epochs}")
                start_time = time.time()
                for x, y in zip(X, Y):
                    # Forward pass
                    output = x
                    activations = []
                    for layer in self.layers:
                        output = layer.forward(output)
                        # Collect activations for debugging
                        if hasattr(layer, 'forward_output'):
                            activations.append(layer.forward_output)

                    loss = self.CCE_loss(y, output)
                    # Compute gradient for the output layer
                    gradient = [y_hat - y_true for y_hat, y_true in zip(output, y)]

                    # Backward pass
                    gradients = [gradient]
                    for layer in reversed(self.layers):
                        gradient = layer.backward(gradient)
                        gradients.append(gradient)
                    for layer in self.layers:
                        if hasattr(layer, 'weights') and layer.weights is not None:
                            layer_weight_shape = Layer.shape(layer.weights)
                            Sequential.apply_deltas(
                                layer.weights, layer.biases, layer.weight_deltas,
                                layer.bias_deltas, self.learning_rate, layer_weight_shape
                            )
                    if i % 10 == 0:
                        print(f"\nIteration {i}, Loss: {loss:.6f}")
                        # Print activation stats
                        for idx, activation in enumerate(activations):
                            min_act, max_act, mean_act = compute_stats(activation)
                            print(f"Layer {idx} ({self.layers[idx].__class__.__name__}) activations - min: {min_act:.6f}, max: {max_act:.6f}, mean: {mean_act:.6f}")

                        # Print gradient stats
                        reversed_layers = list(reversed(self.layers))
                        for idx, grad in enumerate(gradients[:-1]):
                            min_grad, max_grad, mean_grad = compute_stats(grad)
                            layer_name = reversed_layers[idx].__class__.__name__
                            print(f"Layer {len(self.layers)-1-idx} ({layer_name}) gradients - min: {min_grad:.6f}, max: {max_grad:.6f}, mean: {mean_grad:.6f}")

                        # Print weight update stats
                        for idx, layer in enumerate(self.layers):
                            if hasattr(layer, 'weights') and layer.weights is not None:
                                min_w, max_w, mean_w = compute_stats(layer.weights)
                                min_wd, max_wd, mean_wd = compute_stats(layer.weight_deltas)
                                print(f"Layer {idx} ({layer.__class__.__name__}) weights - min: {min_w:.6f}, max: {max_w:.6f}, mean: {mean_w:.6f}")
                                print(f"Layer {idx} ({layer.__class__.__name__}) weight deltas - min: {min_wd:.6f}, max: {max_wd:.6f}, mean: {mean_wd:.6f}")

                    i += 1
                end_time = time.time()
                print(f"Epoch {epoch_index + 1} completed in {end_time - start_time:.2f}s")
        else:
            raise Exception("Unknown optimizer")