import numpy as np
np.random.seed(33)

# =====================
# 1. RNN IMPLEMENTATION
# =====================

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights: small random init
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01   # input -> hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden -> hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden -> output

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        inputs: list of column vectors of shape (input_size, 1)
        returns:
            ys: dict of logits at each time step
            hs: dict of hidden states at each time step (including h[-1])
        """
        hs = {}
        ys = {}
        hs[-1] = np.zeros((self.hidden_size, 1))

        for t in range(len(inputs)):
            z_h = np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            hs[t] = np.tanh(z_h)
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # logits

        self.last_inputs = inputs
        self.last_hs = hs
        return ys, hs

    def backward(self, targets, outputs, hs, learning_rate=1e-2):
        """
        targets: list of integer class indices (length = T)
        outputs: dict of logits from forward()
        hs: dict of hidden states from forward()
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hs[0])
        T = len(self.last_inputs)

        loss = 0.0

        for t in reversed(range(T)):
            logits = outputs[t]                    # (vocab_size, 1)
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)

            # Cross-entropy loss for this step
            loss_t = -np.log(probs[targets[t], 0] + 1e-7)
            loss += loss_t

            # Gradient on logits: probs - one_hot(target)
            dy = probs
            dy[targets[t]] -= 1

            # Gradients for output layer
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # Backprop into hidden state
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh  # tanh derivative

            dbh += dh_raw
            dWxh += np.dot(dh_raw, self.last_inputs[t].T)
            dWhh += np.dot(dh_raw, hs[t - 1].T)

            dh_next = np.dot(self.Whh.T, dh_raw)

        # Average loss per time step
        loss /= T

        # Gradient clipping
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Parameter update
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh  -= learning_rate * dbh
        self.by  -= learning_rate * dby

        return loss

# ============================
# 2. SIMPLE UTILITY COMPONENTS
# ============================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


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

        return -np.log(correct_confidences)

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combined Softmax + Cross-Entropy for faster backward.
    """
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

# =================
# 3. LSTM CELL ONLY
# =================

class LSTM_Cell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # One big weight matrix for all 4 gates: f, i, c~, o
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        x: (input_size, 1)
        h_prev, c_prev: (hidden_size, 1)
        returns:
            h_next, c_next, cache
        """
        combined = np.vstack((h_prev, x))
        gates = np.dot(self.W, combined) + self.b

        f = sigmoid(gates[0:self.hidden_size])
        i = sigmoid(gates[self.hidden_size:2*self.hidden_size])
        c_tilde = np.tanh(gates[2*self.hidden_size:3*self.hidden_size])
        o = sigmoid(gates[3*self.hidden_size:4*self.hidden_size])

        c_next = f * c_prev + i * c_tilde
        h_next = o * np.tanh(c_next)

        cache = (f, i, c_tilde, o, combined, c_prev, c_next)
        return h_next, c_next, cache

# =====================================
# 4. TEXT + VOCAB FOR "NEURAL NETWORK"
# =====================================

text = "neural network"

# Build vocabulary of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

print("Vocabulary:", chars)
print("Vocab size:", vocab_size)

def one_hot(index, vocab_size):
    v = np.zeros((vocab_size, 1))
    v[index] = 1.0
    return v

# Build input and target sequences from "neural network"
inputs = []
targets = []

for t in range(len(text) - 1):
    ch_in = text[t]
    ch_out = text[t + 1]

    x_idx = char_to_idx[ch_in]
    y_idx = char_to_idx[ch_out]

    inputs.append(one_hot(x_idx, vocab_size))
    targets.append(y_idx)

print("Sequence length:", len(inputs))

# ===========================
# 5. SIMPLE RNN TRAINING LOOP
# ===========================

rnn_hidden_size = 32
rnn = RNN(input_size=vocab_size, hidden_size=rnn_hidden_size, output_size=vocab_size)

n_epochs = 200  # small for demo
learning_rate = 0.1

for epoch in range(1, n_epochs + 1):
    outputs, hs = rnn.forward(inputs)
    loss = rnn.backward(targets, outputs, hs, learning_rate=learning_rate)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} - Loss: {loss:.4f}")

# ==============================
# 6. RNN PREDICTION AFTER TRAIN
# ==============================

print("\nRNN predictions for 'neural network' (single pass):")
outputs, hs = rnn.forward(inputs)
for t in range(len(inputs)):
    logits = outputs[t]                   # (vocab_size, 1)
    exp_scores = np.exp(logits - np.max(logits))
    probs = exp_scores / np.sum(exp_scores)
    idx = np.argmax(probs)
    print(f"Input: '{text[t]}' -> Predicted next: '{idx_to_char[idx]}'")

# ==========================
# 7. ONE PASS WITH LSTM CELL
# ==========================

hidden_size = 32
lstm = LSTM_Cell(vocab_size, hidden_size)
output_layer = Layer_Dense(hidden_size, vocab_size)

h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

print("\nLSTM (untrained) predictions for 'neural network':")
for t in range(len(inputs)):
    h, c, cache = lstm.forward(inputs[t], h, c)
    logits = output_layer.forward(h.T)     # (1, hidden) -> (1, vocab)
    exp_values = np.exp(logits - np.max(logits))
    probs = exp_values / np.sum(exp_values)
    idx = np.argmax(probs)
    print(f"Input: '{text[t]}' -> Predicted next: '{idx_to_char[idx]}'")
