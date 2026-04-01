import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class GRU:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hidden_size = hidden_size
        self.lr = lr

        self.Wz = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wr = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_inputs = inputs
        self.last_h = []

        for x in inputs:
            x = np.array([[x]])
            concat = np.vstack((h, x))

            z = sigmoid(self.Wz @ concat + self.bz)   # update gate
            r = sigmoid(self.Wr @ concat + self.br)   # reset gate

            concat_reset = np.vstack((r * h, x))
            h_tilde = tanh(self.Wh @ concat_reset + self.bh)

            h = (1 - z) * h + z * h_tilde

            self.last_h.append((h, z, r, h_tilde, concat, concat_reset))

        y = self.Wy @ h + self.by
        return y

    def backward(self, d_y):
        dWy = d_y @ self.last_h[-1][0].T
        dby = d_y

        d_h = self.Wy.T @ d_y

        dWz = np.zeros_like(self.Wz)
        dWr = np.zeros_like(self.Wr)
        dWh = np.zeros_like(self.Wh)
        dbz = np.zeros_like(self.bz)
        dbr = np.zeros_like(self.br)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(len(self.last_h))):
            h, z, r, h_tilde, concat, concat_reset = self.last_h[t]

            dh_tilde = d_h * z * (1 - h_tilde**2)
            dz = d_h * (h_tilde - h) * z * (1 - z)
            dr = (self.Wh[:, :self.hidden_size].T @ dh_tilde) * h * r * (1 - r)

            dWh += dh_tilde @ concat_reset.T
            dbh += dh_tilde

            dWz += dz @ concat.T
            dbz += dz

            dWr += dr @ concat.T
            dbr += dr

            d_h = (1 - z) * d_h

        for param, dparam in zip(
            [self.Wz, self.Wr, self.Wh, self.Wy, self.bz, self.br, self.bh, self.by],
            [dWz, dWr, dWh, dWy, dbz, dbr, dbh, dby]
        ):
            param -= self.lr * dparam

def generate_data():
    X, y = [], []
    for i in range(1, 50):
        X.append([i, i+1, i+2])
        y.append(i+3)
    return X, y

X, y = generate_data()

model = GRU(input_size=1, hidden_size=16, output_size=1)

epochs = 240

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        seq = X[i]
        target = np.array([[y[i]]])

        out = model.forward(seq)

        loss = (out - target)**2
        total_loss += loss

        d_y = 2 * (out - target)
        model.backward(d_y)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0][0]:.4f}")

test_seq = [10, 11, 12]
pred = model.forward(test_seq)

print("\nInput:", test_seq)
print("Predicted:", pred[0][0])
print("Expected:", 13)