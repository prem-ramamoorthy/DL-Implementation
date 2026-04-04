import numpy as np

def tanh(x):
    return np.tanh(x)

def clip(x, limit=1.0):
    return np.clip(x, -limit, limit)

data_scale = 100.0

class Encoder:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))

        for x in inputs:
            x = np.array([[x]])
            h = tanh(self.Wxh @ x + self.Whh @ h + self.bh)

        return h

class Decoder:
    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, output_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(output_size, hidden_size) * 0.1

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, context, target_len):
        h = context
        outputs = []
        hidden_states = []

        x = np.zeros((1, 1))

        for _ in range(target_len):
            h = tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            hidden_states.append(h)
            y = self.Why @ h + self.by

            outputs.append(y)
            x = y

        return outputs, hidden_states

class Seq2Seq:
    def __init__(self, input_size, hidden_size, output_size):
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, inputs, target_len):
        context = self.encoder.forward(inputs)
        outputs, hidden_states = self.decoder.forward(context, target_len)
        return outputs, hidden_states

def generate_data():
    X, Y = [], []
    for i in range(1, 50):
        X.append([i / data_scale, (i + 1) / data_scale, (i + 2) / data_scale])
        Y.append([(i + 1) / data_scale, (i + 2) / data_scale, (i + 3) / data_scale])
    return X, Y

X, Y = generate_data()

model = Seq2Seq(input_size=1, hidden_size=16, output_size=1)

lr = 0.001
epochs = 200
grad_clip = 1.0

for epoch in range(epochs):
    total_loss = 0

    for i in range(len(X)):
        inputs = X[i]
        targets = Y[i]

        outputs, hidden_states = model.forward(inputs, len(targets))

        loss = 0
        for t in range(len(targets)):
            target = np.array([[targets[t]]])
            loss += float(np.sum((outputs[t] - target) ** 2))

        total_loss += loss

        for t in range(len(targets)):
            d_y = clip(2 * (outputs[t] - np.array([[targets[t]]])), grad_clip)
            h = hidden_states[t]

            model.decoder.Why -= lr * (d_y @ h.T)
            model.decoder.by -= lr * d_y

            model.decoder.Why = clip(model.decoder.Why, 5.0)
            model.decoder.by = clip(model.decoder.by, 5.0)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

test_seq = [5 / data_scale, 6 / data_scale, 7 / data_scale]
outputs, _ = model.forward(test_seq, 3)

print("\nInput:", [5, 6, 7])
print("Predicted sequence:", [o[0][0] * data_scale for o in outputs])
print("Expected:", [6, 7, 8])