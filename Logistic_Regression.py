import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegression:
    def __init__(self, learning_rate: float= 0.01, epochs: int = 100):
        self.W = None
        self.lr = learning_rate
        self.epochs = epochs

    def sigmoid(self, X: np.ndarray):
        logits = X @ self.W
        logits = np.clip(logits, -250, 250)
        logits = 1 / (1 + np.exp(-logits))

        return logits

    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.c_[np.ones(X.shape[0]), X]

        N, D = X.shape
        self.W = np.zeros((D, 1))
        self.losses = []

        for epoch in range(self.epochs):
            y_hat = self.sigmoid(X)

            loss_value = self.loss_fn(y_hat, y)
            self.losses.append(loss_value)

            grad = (y_hat - y).T @ X
            self.W = self.W - 1/N * grad.T * self.lr

    def predict(self, X: np.ndarray):
        X = np.c_[np.ones(X.shape[0]), X]
        logits = self.sigmoid(X)
        predicted_y = np.where(logits >= 0.5, 1, 0)
        return predicted_y

    def evaluate(self, X, y):
        y_hat = self.predict(X)

        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)

        return {
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
        }