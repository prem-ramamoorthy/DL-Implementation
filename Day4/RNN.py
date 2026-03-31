import numpy as np

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x)**2

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.hidden_size = hidden_size
        self.lr = lr

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(output_size, hidden_size) * 0.1

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_hs = { -1: h }

        for t, x in enumerate(inputs):
            x = np.array([[x]])
            h = tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[t] = h

        y = self.Why @ h + self.by
        return y, h

    def backward(self, d_y):
        n = len(self.last_inputs)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        d_h = self.Why.T @ d_y

        for t in reversed(range(n)):
            h = self.last_hs[t]
            h_prev = self.last_hs[t-1]

            temp = dtanh(h) * d_h

            dbh += temp
            dWxh += temp @ np.array([[self.last_inputs[t]]]).T
            dWhh += temp @ h_prev.T

            d_h = self.Whh.T @ temp

        dWhy += d_y @ self.last_hs[n-1].T
        dby += d_y

        for param, dparam in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby]
        ):
            param -= self.lr * dparam

def generate_data(seq_length=3):
    X = []
    y = []
    for i in range(1, 50):
        X.append([i, i+1, i+2])
        y.append(i+3)
    return X, y

X, y = generate_data()

rnn = SimpleRNN(input_size=1, hidden_size=16, output_size=1, lr=0.001)

epochs = 300

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        inputs = X[i]
        target = np.array([[y[i]]])

        out, _ = rnn.forward(inputs)

        loss = np.square(out - target)
        total_loss += loss

        d_y = 2 * (out - target)
        rnn.backward(d_y)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0][0]:.4f}")

test_seq = [10, 11, 12]
pred, _ = rnn.forward(test_seq)

print("\nInput:", test_seq)
print("Predicted next value:", pred[0][0])
print("Expected:", 13)