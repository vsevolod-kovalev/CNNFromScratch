import pickle
from Layer import *

class Sequential:
    def __init__(self, layers: List['Layer']):
        self.layers = layers
        self.optimizer = None
        self.loss = None
        self.learning_rate = DEFAULT_LEARNING_RATE

    @staticmethod
    def save_model_with_pickle(model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved.")

    @staticmethod
    def load_model_from_pickle(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded.")
        return model

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

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def CCE_loss(self, Y, Y_hat) -> float:
        loss = 0.0
        for y, y_hat in zip(Y, Y_hat):
            # Prevent log(0) by adding a small epsilon
            y_hat_safe = y_hat + LOG_EPSILON
            loss += y * math.log(y_hat_safe)
        return -loss

    def compute_stats(self, tensor):
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
                    if i % 50 == 0:
                        Sequential.save_model_with_pickle(self, "model.pkl")
                        print(f"\nIteration {i}, Loss: {loss:.6f}")
                        # Print activation stats
                        for idx, activation in enumerate(activations):
                            min_act, max_act, mean_act = self.compute_stats(activation)
                            print(f"Layer {idx} ({self.layers[idx].__class__.__name__}) activations - min: {min_act:.6f}, max: {max_act:.6f}, mean: {mean_act:.6f}")

                        # Print gradient stats
                        reversed_layers = list(reversed(self.layers))
                        for idx, grad in enumerate(gradients[:-1]):
                            min_grad, max_grad, mean_grad = self.compute_stats(grad)
                            layer_name = reversed_layers[idx].__class__.__name__
                            print(f"Layer {len(self.layers)-1-idx} ({layer_name}) gradients - min: {min_grad:.6f}, max: {max_grad:.6f}, mean: {mean_grad:.6f}")

                        # Print weight update stats
                        for idx, layer in enumerate(self.layers):
                            if hasattr(layer, 'weights') and layer.weights is not None:
                                min_w, max_w, mean_w = self.compute_stats(layer.weights)
                                min_wd, max_wd, mean_wd = self.compute_stats(layer.weight_deltas)
                                print(f"Layer {idx} ({layer.__class__.__name__}) weights - min: {min_w:.6f}, max: {max_w:.6f}, mean: {mean_w:.6f}")
                                print(f"Layer {idx} ({layer.__class__.__name__}) weight deltas - min: {min_wd:.6f}, max: {max_wd:.6f}, mean: {mean_wd:.6f}")

                    i += 1
                end_time = time.time()
                print(f"Epoch {epoch_index + 1} completed in {end_time - start_time:.2f}s")
        else:
            raise Exception("Unknown optimizer")