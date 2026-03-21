import sklearn.neural_network as nn
import sklearn.datasets as ds
import sklearn.model_selection as ms
import matplotlib.pyplot as plt

X, Y = ds.make_moons(n_samples=1000, noise=0.1)
X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.2, random_state=42)

model = nn.MLPClassifier(
        hidden_layer_sizes=(100, 100, 100), 
        max_iter=1000, activation='relu', 
        solver='adam', random_state=42
    )

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, cmap='viridis')
plt.title('MLP Classifier on Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()