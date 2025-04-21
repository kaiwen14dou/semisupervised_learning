import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def train_manual_logistic_regression(
    df,
    outcome_col="Y",
    feature_start_col=5,
    cutoff=14,
    test_size=0.2,
    pos_weight=0.8,
    neg_weight=0.2,
    lr= 0.01,
    epochs = 30,
    verbose=True
):

    class ManualLogisticRegression(nn.Module):
        def __init__(self, input_dim, pos_weight=0.8, neg_weight=0.2):
            super().__init__()
            self.w = nn.Parameter(torch.randn(input_dim, 1, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.pos_weight = pos_weight
            self.neg_weight = neg_weight

        def forward(self, x):
            logits = x @ self.w + self.b
            return torch.sigmoid(logits)

        def loss(self, output, target):
            output = torch.clamp(output, min=1e-8, max=1 - 1e-8) 
            loss = - (pos_weight * target * torch.log(output) +
              neg_weight * (1 - target) * torch.log(1 - output))
            return loss.mean()

        def predict_proba(self, x):
            self.eval()
            with torch.no_grad():
                return self.forward(x)

        def predict(self, x, threshold=0.5):
            return (self.predict_proba(x) >= threshold).float()

    y_raw = df[outcome_col]
    y_binary = (y_raw > cutoff).astype(int)
    x = df.iloc[:, feature_start_col:]

    X_train, X_test, y_train, y_test = train_test_split(x, y_binary, test_size=test_size, stratify=y_binary, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    model = ManualLogisticRegression(input_dim=X_train.shape[1], pos_weight=pos_weight, neg_weight=neg_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs = model(X_train_tensor)
        loss = model.loss(probs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_test_tensor).squeeze().numpy()
        y_true = y_test_tensor.squeeze().numpy()

    auc = roc_auc_score(y_true, probs)
    logloss = -2 * np.mean(
        pos_weight * y_true * np.log(np.clip(probs, 1e-8, 1)) +
        neg_weight * (1 - y_true) * np.log(np.clip(1 - probs, 1e-8, 1))
    )

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Manual Logistic Regression ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"\n✅ Final AUC: {auc:.3f}")
        print(f"✅ Final Weighted LogLoss: {logloss:.3f}")

    return model, auc, logloss