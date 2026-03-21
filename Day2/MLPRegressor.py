import numpy as np

class MLPRegressorFromScratch:
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size=1,
        learning_rate=0.01,
        epochs=5000,
        seed=42
    ):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            self.weights.append(np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim))
            self.biases.append(np.zeros((1, out_dim)))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        activations = [X]
        zs = []

        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            zs.append(Z)
            activations.append(A)

        Z_out = A @ self.weights[-1] + self.biases[-1]
        zs.append(Z_out)
        activations.append(Z_out)

        return zs, activations

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y, zs, activations):
        m = y.shape[0]

        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        dA = 2 * (activations[-1] - y) / m
        dZ = dA

        dW[-1] = activations[-2].T @ dZ
        db[-1] = np.sum(dZ, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            dA = dZ @ self.weights[i + 1].T
            dZ = dA * self.relu_derivative(zs[i])

            dW[i] = activations[i].T @ dZ
            db[i] = np.sum(dZ, axis=0, keepdims=True)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y, print_every=500):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for epoch in range(1, self.epochs + 1):
            zs, activations = self.forward(X)
            y_pred = activations[-1]
            loss = self.mse_loss(y, y_pred)
            self.backward(y, zs, activations)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        _, activations = self.forward(X)
        return activations[-1]


X = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = np.array([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
])

model = MLPRegressorFromScratch(
    input_size=1,
    hidden_sizes=[8, 8],
    output_size=1,
    learning_rate=0.001,
    epochs=10000
)

model.fit(X, y, print_every=1000)

predictions = model.predict(X)
print("\nPredictions:")
print(predictions)