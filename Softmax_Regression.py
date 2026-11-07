import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


class SoftmaxRegression:

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 400,
        num_classes: int = 10,
    ):
        self.W = None
        self.lr = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.losses = []

    def softmax(self, X: np.ndarray):
        # Nhân ma trận trên kiểu float32
        logits = X @ self.W
        exp_logits = np.exp(
            logits - np.max(logits, axis=1, keepdims=True), dtype=np.float32
        )
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def one_hot(self, y: np.ndarray):
        y = y.ravel()
        Y_oh = np.zeros((y.size, self.num_classes), dtype=np.float32)
        Y_oh[np.arange(y.size), y] = 1.0
        return Y_oh

    def loss_fn(self, y_hat: np.ndarray, Y_oh: np.ndarray):
        eps = 1e-7
        y_hat = np.clip(y_hat, eps, 1.0 - eps)
        return -np.mean(np.sum(Y_oh * np.log(y_hat), axis=1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Tạo cột bias kiểu float32
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        X = np.hstack([ones, X])

        N, D = X.shape
        # Khởi tạo W BẮT BUỘC là float32
        self.W = np.zeros((D, self.num_classes), dtype=np.float32)
        self.losses = []

        Y_oh = self.one_hot(y)

        for epoch in range(self.epochs):
            y_hat = self.softmax(X)

            loss_value = self.loss_fn(y_hat, Y_oh)
            self.losses.append(loss_value)

            # Gradient Descent bằng float32
            grad = (np.float32(1.0) / N) * (X.T @ (y_hat - Y_oh))
            self.W -= np.float32(self.lr) * grad

    def predict(self, X: np.ndarray):
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        X = np.hstack([ones, X])
        probs = self.softmax(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, target_names=None):
        y_hat = self.predict(X)
        y = y.ravel()

        precision = precision_score(y, y_hat, average="macro")
        recall = recall_score(y, y_hat, average="macro")
        f1 = f1_score(y, y_hat, average="macro")
        report = classification_report(y, y_hat, target_names=target_names, zero_division=0)

        return {
            "Precision (Macro)": precision,
            "Recall (Macro)": recall,
            "F1-score (Macro)": f1,
            "Classification Report": report,
        }