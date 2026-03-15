class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            for i in range(len(X)):
                linear_output = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                y_pred = self.step_function(linear_output)

                error = y[i] - y_pred

                for j in range(n_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error

                print(
                    f"Input: {X[i]}, Target: {y[i]}, Predicted: {y_pred}, "
                    f"Error: {error}, Weights: {self.weights}, Bias: {self.bias}"
                )
            print("-" * 50)

    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias
            y_pred = self.step_function(linear_output)
            predictions.append(y_pred)
        return predictions

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 0, 0, 1]

perceptron = SingleLayerPerceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

print("Final Weights:", perceptron.weights)
print("Final Bias:", perceptron.bias)

test_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

predictions = perceptron.predict(test_data)
print("Predictions:", predictions)