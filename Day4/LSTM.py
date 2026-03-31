import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        def init_gate():
            return np.random.randn(hidden_size, hidden_size + input_size) * 0.1
        
        self.Wf = init_gate()
        self.Wi = init_gate()
        self.Wc = init_gate()
        self.Wo = init_gate()
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.h, self.c = {}, {}
        self.f, self.i, self.c_bar, self.o = {}, {}, {}, {}
        self.x = inputs
        
        self.h[-1] = np.zeros((self.hidden_size, 1))
        self.c[-1] = np.zeros((self.hidden_size, 1))
        
        for t in range(len(inputs)):
            concat = np.vstack((self.h[t-1], inputs[t]))
            
            self.f[t] = sigmoid(self.Wf @ concat + self.bf)
            self.i[t] = sigmoid(self.Wi @ concat + self.bi)
            self.c_bar[t] = tanh(self.Wc @ concat + self.bc)
            
            self.c[t] = self.f[t] * self.c[t-1] + self.i[t] * self.c_bar[t]
            
            self.o[t] = sigmoid(self.Wo @ concat + self.bo)
            self.h[t] = self.o[t] * tanh(self.c[t])
        
        y = self.Wy @ self.h[len(inputs)-1] + self.by
        return y

    def backward(self, target, output, lr=0.01):
        dy = output - target
        
        dWy = dy @ self.h[len(self.x)-1].T
        dby = dy
        
        dh = self.Wy.T @ dy
        dc = np.zeros_like(self.c[0])
        
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)
        
        for t in reversed(range(len(self.x))):
            o = self.o[t]
            c = self.c[t]
            c_prev = self.c[t-1]
            h_prev = self.h[t-1]
            
            tanh_c = tanh(c)
            
            do = dh * tanh_c * o * (1 - o)
            
            dc = dh * o * (1 - tanh_c**2) + dc
            
            df = dc * c_prev * self.f[t] * (1 - self.f[t])
            di = dc * self.c_bar[t] * self.i[t] * (1 - self.i[t])
            dc_bar = dc * self.i[t] * (1 - self.c_bar[t]**2)
            
            concat = np.vstack((h_prev, self.x[t]))
            
            dWf += df @ concat.T
            dWi += di @ concat.T
            dWc += dc_bar @ concat.T
            dWo += do @ concat.T
            
            dbf += df
            dbi += di
            dbc += dc_bar
            dbo += do
            
            dconcat = (
                self.Wf.T @ df +
                self.Wi.T @ di +
                self.Wc.T @ dc_bar +
                self.Wo.T @ do
            )
            
            dh = dconcat[:self.hidden_size]
            dc = dc * self.f[t]
        
        for param, dparam in zip(
            [self.Wf, self.Wi, self.Wc, self.Wo, self.bf, self.bi, self.bc, self.bo, self.Wy, self.by],
            [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWy, dby]
        ):
            param -= lr * dparam
            
def create_data(seq_len=4, num_samples=100):
    X, Y = [], []
    for i in range(num_samples):
        start = np.random.randint(0, 50)
        seq = [start + j for j in range(seq_len+1)]
        
        X.append([np.array([[x]]) for x in seq[:-1]])
        Y.append(np.array([[seq[-1]]]))
    return X, Y

np.random.seed(1)
lstm = LSTM(input_size=1, hidden_size=16, output_size=1)

X, Y = create_data()

for epoch in range(200):
    loss = 0
    for x, y in zip(X, Y):
        out = lstm.forward(x)
        loss += np.mean((out - y)**2)
        lstm.backward(y, out, lr=0.001)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
test_seq = [np.array([[10]]), np.array([[11]]), np.array([[12]]), np.array([[13]])]
pred = lstm.forward(test_seq)

print("Prediction:", pred)