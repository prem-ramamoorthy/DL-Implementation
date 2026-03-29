import numpy as np

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy_loss(pred, y):
    m = y.shape[0]
    return -np.sum(y * np.log(pred + 1e-8)) / m

class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input):
        self.input = input
        h, w = input.shape
        f = self.filter_size
        output = np.zeros((self.num_filters, h - f + 1, w - f + 1))

        for k in range(self.num_filters):
            for i in range(h - f + 1):
                for j in range(w - f + 1):
                    region = input[i:i+f, j:j+f]
                    output[k, i, j] = np.sum(region * self.filters[k])
        return output

    def backward(self, dL_dout, lr):
        dL_dfilters = np.zeros_like(self.filters)

        for k in range(self.num_filters):
            for i in range(dL_dout.shape[1]):
                for j in range(dL_dout.shape[2]):
                    region = self.input[i:i+self.filter_size, j:j+self.filter_size]
                    dL_dfilters[k] += dL_dout[k, i, j] * region

        self.filters -= lr * dL_dfilters

class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, dL_dout):
        return dL_dout * (self.input > 0)

class MaxPool:
    def __init__(self, size):
        self.size = size

    def forward(self, input):
        self.input = input
        c, h, w = input.shape
        s = self.size

        output = np.zeros((c, h//s, w//s))

        for k in range(c):
            for i in range(0, h, s):
                for j in range(0, w, s):
                    region = input[k, i:i+s, j:j+s]
                    output[k, i//s, j//s] = np.max(region)

        return output

    def backward(self, dL_dout):
        c, h, w = self.input.shape
        s = self.size
        dL_dinput = np.zeros_like(self.input)

        for k in range(c):
            for i in range(0, h, s):
                for j in range(0, w, s):
                    region = self.input[k, i:i+s, j:j+s]
                    max_val = np.max(region)

                    for x in range(s):
                        for y in range(s):
                            if region[x, y] == max_val:
                                dL_dinput[k, i+x, j+y] = dL_dout[k, i//s, j//s]

        return dL_dinput

class Flatten:
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten().reshape(1, -1)

    def backward(self, dL_dout):
        return dL_dout.reshape(self.input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, dL_dout, lr):
        dL_dW = np.dot(self.input.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0, keepdims=True)
        dL_dinput = np.dot(dL_dout, self.weights.T)

        self.weights -= lr * dL_dW
        self.bias -= lr * dL_db

        return dL_dinput

class CNN:
    def __init__(self):
        self.conv = Conv2D(2, 3)
        self.relu = ReLU()
        self.pool = MaxPool(2)
        self.flatten = Flatten()
        self.fc = Dense(2 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.relu.forward(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return softmax(x)

    def backward(self, pred, y, lr):
        m = y.shape[0]
        grad = (pred - y) / m

        grad = self.fc.backward(grad, lr)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        self.conv.backward(grad, lr)

if __name__ == "__main__":
    np.random.seed(0)

    model = CNN()

    X = np.random.randn(28, 28)
    y = np.zeros((1, 10))
    y[0, np.random.randint(0, 10)] = 1
    
    print("X" , X.shape)
    print("y" , y)

    for epoch in range(10):
        pred = model.forward(X)
        loss = cross_entropy_loss(pred, y)

        model.backward(pred, y, lr=0.01)

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        
    x_pred = model.forward(np.random.randn(28, 28))
    print("Predicted probabilities:", x_pred)