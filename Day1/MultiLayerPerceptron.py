import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1, seed=42):
        np.random.seed(seed)

        self.learning_rate = learning_rate
        self.layers = [input_size] + hidden_sizes + [output_size]
        print(f"Layer sizes: {self.layers}")
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            in_size = self.layers[i]
            out_size = self.layers[i + 1]

            w = np.random.randn(in_size, out_size) * np.sqrt(1 / in_size) # Xavier initialization multiplied by sqrt(1/in_size) to keep variance of activations stable
            b = np.zeros((1, out_size))

            self.weights.append(w)
            self.biases.append(b)
        print("Initial weights and biases:")
        print("Weights:")
        for w in self.weights:
            print(w)
        print("Biases:")
        for b in self.biases:
            print(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        activations = [X] # Store input as the first activation for backpropagation
        z_values = []

        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)

            z_values.append(z)
            activations.append(a)

        return z_values, activations

    def compute_loss(self, y_true, y_pred): # Binary cross-entropy loss
        # L = -1/m * sum(y_i * log(y_pred_i) + (1 - y_i) * log(1 - y_pred_i))
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y, z_values, activations):
        m = X.shape[0]

        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)

        delta = activations[-1] - y 

        for i in reversed(range(len(self.weights))):
            d_weights[i] = np.dot(activations[i].T, delta) / m
            d_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(self, X, y, epochs=10000, print_every=1000):
        for epoch in range(1, epochs + 1):
            z_values, activations = self.forward(X)
            y_pred = activations[-1]

            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, z_values, activations)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        _, activations = self.forward(X)
        return (activations[-1] >= 0.5).astype(int)

    def predict_proba(self, X):
        _, activations = self.forward(X)
        return activations[-1]

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

mlp = MLP(input_size=2, hidden_sizes=[4, 4], output_size=1, learning_rate=0.5)

mlp.train(X, y, epochs=10000, print_every=1000)

print("\nPredicted probabilities:")
print(mlp.predict_proba(X))

print("\nPredicted classes:")
print(mlp.predict(X))