import numpy as np
np.random.seed(33) 

class Layer_Conv:
    def __init__(self, n_filters, kernel_size, stride=1, padding=0):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He-like initialization
        self.filters = np.random.randn(n_filters, kernel_size, kernel_size) / (kernel_size ** 2)
        self.biases = np.zeros(n_filters)

    def _pad(self, x):
        if self.padding == 0:
            return x
        return np.pad(x, ((self.padding, self.padding), (self.padding, self.padding)), mode="constant")

    def forward(self, input):
        self.last_input = input
        x = self._pad(input)
        h, w = x.shape

        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        out = np.zeros((self.n_filters, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                i0 = i * self.stride
                j0 = j * self.stride
                region = x[i0:i0 + self.kernel_size, j0:j0 + self.kernel_size]
                out[:, i, j] = np.sum(region * self.filters, axis=(1, 2)) + self.biases

        self.output = out
        return out

    def backward(self, d_l_d_out, learning_rate):
        d_l_d_filters = np.zeros_like(self.filters)
        d_l_d_biases = np.zeros_like(self.biases)

        x = self._pad(self.last_input)
        h, w = x.shape
        out_h, out_w = d_l_d_out.shape[1], d_l_d_out.shape[2]

        for i in range(out_h):
            for j in range(out_w):
                i0 = i * self.stride
                j0 = j * self.stride
                region = x[i0:i0 + self.kernel_size, j0:j0 + self.kernel_size]

                for f in range(self.n_filters):
                    d_l_d_filters[f] += d_l_d_out[f, i, j] * region
                    d_l_d_biases[f] += d_l_d_out[f, i, j]

        self.filters -= learning_rate * d_l_d_filters
        self.biases -= learning_rate * d_l_d_biases
        return None
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues, learning_rate):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
class MaxPool2:
    def forward(self, input):
        self.last_input = input
        num_filters, h, w = input.shape
        out = np.zeros((num_filters, h // 2, w // 2))

        for i in range(h // 2):
            for j in range(w // 2):
                region = input[:, i*2:i*2+2, j*2:j*2+2]
                out[:, i, j] = np.max(region, axis=(1, 2))

        self.output = out
        return out

    def backward(self, d_l_d_out):
        num_filters, h, w = self.last_input.shape
        d_l_d_input = np.zeros_like(self.last_input)

        for i in range(h // 2):
            for j in range(w // 2):
                region = self.last_input[:, i*2:i*2+2, j*2:j*2+2]

                for f in range(num_filters):
                    region_f = region[f]
                    idx = np.unravel_index(np.argmax(region_f), region_f.shape)
                    d_l_d_input[f, i*2 + idx[0], j*2 + idx[1]] = d_l_d_out[f, i, j]

        return d_l_d_input
class Layer_Flatten:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(1, -1)

    def backward(self, dvalues):
        return dvalues.reshape(self.input_shape)
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
    
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# --- Initialize network ---
conv = Layer_Conv(8, 3, stride=1, padding=0)
pool = MaxPool2()
flatten = Layer_Flatten()
dense = Layer_Dense(8 * 13 * 13, 10)
loss_softmax = Activation_Softmax_Loss_CategoricalCrossentropy()

learning_rate = 0.01

# Fake 28x28 grayscale image and label
image = np.random.randn(28, 28)
label = 7
y_true = np.array([label])

# --- Forward pass ---
out = conv.forward(image)                  # (8, 26, 26)
out = np.maximum(0, out)                  # ReLU
out = pool.forward(out)                   # (8, 13, 13)
out = flatten.forward(out)                # (1, 1352)
logits = dense.forward(out)               # (1, 10)
loss = loss_softmax.forward(logits, y_true)

print(f"Initial loss: {loss:.4f}")

# --- Backward pass ---
loss_softmax.backward(loss_softmax.output, y_true)
dense.backward(loss_softmax.dinputs, learning_rate=learning_rate)
d_flatten = flatten.backward(dense.dinputs)
d_pool = pool.backward(d_flatten)
conv.backward(d_pool, learning_rate=learning_rate)

print("Backward pass complete. Parameters updated.")
