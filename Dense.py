from Layer import *

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