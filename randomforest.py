import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#
# # 假设文件路径为 /mnt/data/01b9b88e9483ac573e4b3711f9cfa800.xlsx
# file_path = "output/corrected_avg_1.xlsx"
#
# # 读取 Excel 文件
# df = pd.read_excel(file_path)
#
# # 数据预处理：清除非特征列
# non_feature_cols = ['id', 'label']
# feature_cols = df.columns.difference(non_feature_cols)
#
# # 将 label 为 1 或 2 的合并成一个新标签，如 'P'（代表 Positive）
# df['label'] = df['label'].replace({1: 'P', 2: 'P', '1': 'P', '2': 'P', 'N': 'N'})
# df['label'] = df['label'].astype(str)
#
# # 特征和标签分离
# X = df[feature_cols]
# y = LabelEncoder().fit_transform(df['label'])
#
# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Step 4: PCA降维
# pca = PCA(n_components=30)
# X_pca = pca.fit_transform(X_scaled)
#
# print(f"PCA降维后特征维数: {X_pca.shape[1]}")
#
# # 数据划分
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)


# 假设文件路径为 /mnt/data/01b9b88e9483ac573e4b3711f9cfa800.xlsx
file_path = "output/corrected_avg_1.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)

# 数据预处理：清除非特征列
non_feature_cols = ['id', 'label', 'flag0', 'feature_num', 'total_y']
feature_cols = df.columns.difference(non_feature_cols)

# 标签合并处理（处理NP分类）
df['label'] = df['label'].replace({1: 'P', 2: 'P', '1': 'P', '2': 'P', 'N': 'N'})

# 用于多条ROC曲线绘制
# df['label'] = df['label'].replace({1: 'P1', 2: 'P2', '1': 'P1', '2': 'P2', 'N': 'N'})

# 处理亚型分类
# df = df[df['label'] != 'N']
# df['label'] = df['label'].replace({1: '1', 2: '2', '1': '1', '2': '2', 'N': 'N'})

df['label'] = df['label'].astype(str)

# 特征和标签分离
X = df[feature_cols]
y = LabelEncoder().fit_transform(df['label'])  # 标签编码为0和1

# 按 flag0 划分数据
X_train_raw = X[df['flag0'] == 1]  # flag0=1 的样本作为训练集
X_test_raw = X[df['flag0'] == 0]   # flag0=0 的样本作为测试集
y_train = y[df['flag0'] == 1]
y_test = y[df['flag0'] == 0]

# 标准化处理（仅在训练集上拟合）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)  # 训练集拟合并转换
X_test_scaled = scaler.transform(X_test_raw)       # 测试集仅转换

# PCA降维（仅在训练集上拟合）
pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train_scaled)    # 训练集拟合并转换
X_test_pca = pca.transform(X_test_scaled)          # 测试集仅转换

# 定义参数网格（示例值，可调整范围）
param_grid = {
    'n_estimators': [100, 200],       # 决策树的数量
    'max_depth': [None, 10, 20],      # 树的最大深度（None表示不限）
    'max_features': ['sqrt', 'log2'],  # 分裂时考虑的特征数量
    'min_samples_split': [2, 5, 10]   # 分裂节点所需最小样本数
}

# 创建随机森林分类器
rf = RandomForestClassifier(
    random_state=42,          # 固定随机种子
    class_weight='balanced',  # 处理类别不平衡（可选）
    oob_score=True            # 启用袋外（Out-of-Bag）评估
)

# 网格搜索交叉验证
clf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                      # 5折交叉验证
    scoring='roc_auc_ovr',     # 与之前评分一致
    refit=True,                # 用最佳参数重新训练
    n_jobs=-1                  # 并行计算加速
)
clf.fit(X_train_pca, y_train)
print("Best parameters:", clf.best_params_)

# AUC 和 ROC 曲线
y_pred = clf.predict(X_test_pca)
y_proba = clf.predict_proba(X_test_pca)  # 替换 decision_function 为 predict_proba
y_test_bin = label_binarize(y_test, classes=list(set(y_test)))  # 修复：明确使用 y_test 的类别

# 如果是二分类
if y_test_bin.shape[1] == 1:
    y_score = y_proba[:, 1]  # 二分类时取正类概率

    def calculate_metrics(y_true, y_pred, y_score):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'specificity': tn / (tn + fp),
            'auc': roc_auc_score(y_true, y_score)  # 直接使用概率分数
        }


    # ================ 新增：Bootstrap置信区间计算 ================
    def bootstrap_ci(y_true, y_pred, y_score, n_bootstraps=1000):
        metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'specificity': [], 'auc': []
        }

        for _ in range(n_bootstraps):
            # 修复点：直接使用索引，无需 .iloc
            indices = resample(np.arange(len(y_true)), replace=True)
            y_test_boot = y_true[indices]  # 直接索引
            y_pred_boot = y_pred[indices]
            y_score_boot = y_score[indices]

            # 跳过无效采样（如只有单类别）
            if len(np.unique(y_test_boot)) < 2:
                continue

            # 计算指标
            try:
                res = calculate_metrics(
                    y_test_boot,
                    y_pred_boot,
                    y_score_boot
                )
                for k in metrics:
                    metrics[k].append(res[k])
            except:
                continue

        # 计算95%置信区间
        ci = {}
        for k in metrics:
            if metrics[k]:
                ci[k] = (
                    np.percentile(metrics[k], 2.5),
                    np.percentile(metrics[k], 97.5)
                )
        return ci


    # ================ 执行计算 ================
    # 原始指标
    metrics = calculate_metrics(y_test, y_pred, y_score)

    # 置信区间（1000次Bootstrap）
    ci = bootstrap_ci(y_test, y_pred, y_score, n_bootstraps=1000)

    # ================ 可视化增强 ================
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(10, 6))

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC Curve (AUC = {metrics["auc"]:.2f} [{ci["auc"][0]:.2f}-{ci["auc"][1]:.2f}])')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # 添加指标文本框
    text = (
        f'Accuracy: {metrics["accuracy"]:.2f} [{ci["accuracy"][0]:.2f}-{ci["accuracy"][1]:.2f}]\n'
        f'Precision: {metrics["precision"]:.2f} [{ci["precision"][0]:.2f}-{ci["precision"][1]:.2f}]\n'
        f'Recall: {metrics["recall"]:.2f} [{ci["recall"][0]:.2f}-{ci["recall"][1]:.2f}]\n'
        f'F1: {metrics["f1"]:.2f} [{ci["f1"][0]:.2f}-{ci["f1"][1]:.2f}]\n'
        f'Specificity: {metrics["specificity"]:.2f} [{ci["specificity"][0]:.2f}-{ci["specificity"][1]:.2f}]'
    )
    plt.text(0.6, 0.25, text, bbox=dict(facecolor='white', alpha=0.8),
             fontsize=10)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve with 95% CIs')  # 修改标题
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    auc_macro = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    fpr = dict()
    tpr = dict()
    for i in range(y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])  # 使用概率列
        plt.plot(fpr[i], tpr[i], label=f'Class {i} AUC = {roc_auc_score(y_test_bin[:, i], y_proba[:, i]):.2f}')
    plt.title(f'Multiclass ROC Curve (Macro AUC = {auc_macro:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
