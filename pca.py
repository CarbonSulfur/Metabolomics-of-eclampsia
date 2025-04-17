import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 读取数据（如果你已经有一个 CSV 或 Excel 文件）
df = pd.read_excel("output/corrected_avg_1.xlsx")  # 你也可以用 pd.read_excel

# 2. 提取特征列（从第5列开始为数值特征）
X = df.iloc[:, 4:].values  # 假设前4列为 id、label、total_y、feature_num

# 3. 标准化特征
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 4. PCA降维到二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# 5. 将降维结果加入原始 DataFrame 中（便于可视化）
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# 6. 可视化结果（按 label 分颜色）
plt.figure(figsize=(8, 6))
for label in df['label'].unique():
    subset = df[df['label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=label)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
