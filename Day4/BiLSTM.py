import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - x**2

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))

        f = sigmoid(self.Wf @ concat + self.bf)
        i = sigmoid(self.Wi @ concat + self.bi)
        c_bar = tanh(self.Wc @ concat + self.bc)
        o = sigmoid(self.Wo @ concat + self.bo)

        c = f * c_prev + i * c_bar
        h = o * tanh(c)

        cache = (x, h_prev, c_prev, f, i, c_bar, o, c)
        return h, c, cache

class BiLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hidden_size = hidden_size
        self.lr = lr

        self.lstm_fwd = LSTMCell(input_size, hidden_size)
        self.lstm_bwd = LSTMCell(input_size, hidden_size)

        self.Wy = np.random.randn(output_size, hidden_size * 2) * 0.1
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        T = len(inputs)

        h_fwd, c_fwd = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
        self.fwd_states = []

        for t in range(T):
            x = np.array([[inputs[t]]])
            h_fwd, c_fwd, cache = self.lstm_fwd.forward(x, h_fwd, c_fwd)
            self.fwd_states.append((h_fwd, c_fwd, cache))

        h_bwd, c_bwd = np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))
        self.bwd_states = []

        for t in reversed(range(T)):
            x = np.array([[inputs[t]]])
            h_bwd, c_bwd, cache = self.lstm_bwd.forward(x, h_bwd, c_bwd)
            self.bwd_states.insert(0, (h_bwd, c_bwd, cache))

        h_final = np.vstack((self.fwd_states[-1][0], self.bwd_states[0][0]))

        y = self.Wy @ h_final + self.by
        return y

def generate_data():
    X, y = [], []
    for i in range(1, 50):
        X.append([i, i+1, i+2])
        y.append(i+3)
    return X, y

X, y = generate_data()

model = BiLSTM(input_size=1, hidden_size=16, output_size=1)

epochs = 200

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        seq = X[i]
        target = np.array([[y[i]]])

        out = model.forward(seq)

        loss = (out - target)**2
        total_loss += loss

        d_y = 2 * (out - target)

        h_fwd = model.fwd_states[-1][0]
        h_bwd = model.bwd_states[0][0]
        h_final = np.vstack((h_fwd, h_bwd))

        dWy = d_y @ h_final.T
        dby = d_y

        model.Wy -= model.lr * dWy
        model.by -= model.lr * dby

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0][0]:.4f}")

test_seq = [10, 11, 12]
pred = model.forward(test_seq)

print("\nInput:", test_seq)
print("Predicted:", pred[0][0])
print("Expected:", 13)