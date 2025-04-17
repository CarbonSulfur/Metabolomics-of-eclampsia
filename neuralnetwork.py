import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 1. 指标计算函数
def calculate_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp),
        'auc': roc_auc_score(y_true, y_score)
    }

# 2. Bootstrap 置信区间
def bootstrap_ci(y_true, y_pred, y_score, n_bootstraps=1000):
    metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'specificity': [], 'auc': []
    }
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        y_test_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_score_boot = y_score[indices]
        if len(np.unique(y_test_boot)) < 2:
            continue
        try:
            res = calculate_metrics(y_test_boot, y_pred_boot, y_score_boot)
            for k in metrics:
                metrics[k].append(res[k])
        except:
            continue
    ci = {}
    for k in metrics:
        if metrics[k]:
            ci[k] = (
                np.percentile(metrics[k], 2.5),
                np.percentile(metrics[k], 97.5)
            )
    return ci


# 3. 数据加载与预处理
file_path = "output/corrected_avg_1.xlsx"
df = pd.read_excel(file_path)
non_feature_cols = ['id', 'label', 'flag0', 'feature_num', 'total_y']
feature_cols = df.columns.difference(non_feature_cols)
df['label'] = df['label'].replace({1: 'P', 2: 'P', '1': 'P', '2': 'P', 'N': 'N'})
df['label'] = df['label'].astype(str)

X = df[feature_cols]
y = LabelEncoder().fit_transform(df['label'])
X_train_raw = X[df['flag0'] == 1]
X_test_raw = X[df['flag0'] == 0]
y_train = y[df['flag0'] == 1]
y_test = y[df['flag0'] == 0]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
n_features = X_train_pca.shape[1]

# 4. 定义 PyTorch 网络结构
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units=64, dropout_rate=0.2, activation='relu'):
        super(SimpleNN, self).__init__()
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units // 2),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 5. 准备数据加载器
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 6. 模型训练
model = SimpleNN(n_features)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_auc = []
train_loss = []

for epoch in range(100):
    model.train()
    epoch_losses = []
    y_probs, y_trues = [], []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        y_probs.extend(outputs.detach().numpy())
        y_trues.extend(y_batch.numpy())

    y_probs = np.array(y_probs).flatten()
    y_trues = np.array(y_trues).flatten()
    epoch_auc = roc_auc_score(y_trues, y_probs)
    train_auc.append(epoch_auc)
    train_loss.append(np.mean(epoch_losses))

# 7. 测试集预测与评估
model.eval()
with torch.no_grad():
    y_test_proba = model(X_test_tensor).numpy().flatten()
    y_test_pred = (y_test_proba > 0.5).astype(int)

metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
ci = bootstrap_ci(y_test, y_test_pred, y_test_proba)

# 8. 绘图
plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr, label=(f'AUC = {metrics["auc"]:.2f} [{ci["auc"][0]:.2f}-{ci["auc"][1]:.2f}]'))
plt.text(
    x=0.05,  # X坐标（相对图表左侧的位置）
    y=0.95,  # Y坐标（相对图表顶部的位置）
    s=(
        f'Acc = {metrics["accuracy"]:.2f} [{ci["accuracy"][0]:.2f}-{ci["accuracy"][1]:.2f}], \n'
        f'Prec = {metrics["precision"]:.2f} [{ci["precision"][0]:.2f}-{ci["precision"][1]:.2f}]\n'
        f'Recall = {metrics["recall"]:.2f} [{ci["recall"][0]:.2f}-{ci["recall"][1]:.2f}],\n '
        f'F1 = {metrics["f1"]:.2f} [{ci["f1"][0]:.2f}-{ci["f1"][1]:.2f}]'
    ),
    transform=plt.gca().transAxes,  # 使用相对坐标
    verticalalignment='top',        # 文本顶部对齐
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)  # 添加背景框
)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve with 95% CI')
plt.legend()
plt.grid()
plt.show()

# 9. 训练过程可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_auc, label='Train AUC')
plt.title('Training AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

