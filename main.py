import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from match import peak_match
from normalize import normalize

def plot_correlation_heatmap(prepared_datas, corrected_datas):
    def calculate_correlations(datas):
        all_y = []  # 用于存储所有患者的第一组数据的y值
        for patient in datas:
            for data in patient["datas"]:
                x,y = data
                all_y.append(y)
        n = len(all_y)
        # 创建相关系数矩阵
        corr_matrix = np.zeros((n, n))
        # 计算相关系数
        for i in range(n):
            for j in range(n):
                corr = np.corrcoef(all_y[i], all_y[j])[0,1]
                corr_matrix[i,j] = corr
                
        return corr_matrix

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 计算并绘制校正前的相关系数热图
    corr_before = calculate_correlations(prepared_datas)
    im1 = ax1.imshow(corr_before, cmap='viridis')
    ax1.set_title('Correlation Matrix (Before Correction)')

    # 计算并绘制校正后的相关系数热图
    corr_after = calculate_correlations(corrected_datas)
    im2 = ax2.imshow(corr_after, cmap='viridis')
    ax2.set_title('Correlation Matrix (After Correction)')

    # 添加颜色条
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

def save_data(reference_data, corrected_datas):
    def make_path(name = None, ext = None):
        current_path = os.getcwd()
        save_path = os.path.join(current_path, "output")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, name + "." + ext)
        return save_path
    
    # 参考数据的峰值
    all_peaks = np.nonzero(reference_data["datas"][0][1])[0]
    # 所有可能的x值
    all_x = corrected_datas[0]["datas"][0][0]
    # 数据的列名
    columns = ["id","label","total_y","feature_num"]+[str(all_x[peak]) for peak in all_peaks]
    df1 = pd.DataFrame(columns=columns)     # 每个患者的五组数据
    df2 = pd.DataFrame(columns=columns)     # 每个患者的五组数据的和
    df3 = pd.DataFrame(columns=columns)     # 每个患者的五组数据的平均值
    for patient in corrected_datas:
        name = patient["name"]
        for j in range(5):
            x,y = patient["datas"][j]
            row1 = {
                "id": name + "_" + str(j+1),
                "label": None,
                "total_y": np.sum(y[all_peaks]),
                "feature_num": np.sum(y[all_peaks]>0).astype(int)
            }
            row1.update({str(all_x[peak]): y[peak] for peak in all_peaks})
            df1 = pd.concat([df1, pd.DataFrame([row1])], ignore_index=True)
            if(j==0): sum_y = y
            else: sum_y += y
        row2 = {
            "id": name,
            "label": None,
            "total_y": np.sum(sum_y[all_peaks]),
            "feature_num": np.sum(sum_y[all_peaks]>0).astype(int)
        }
        row2.update({str(all_x[peak]): sum_y[peak] for peak in all_peaks})
        df2 = pd.concat([df2, pd.DataFrame([row2])], ignore_index=True)
        row3 = {
            "id": name,
            "label": None,
            "total_y": np.sum(sum_y[all_peaks])/5,
            "feature_num": np.sum(sum_y[all_peaks]>0).astype(int)
        }
        row3.update({str(all_x[peak]): sum_y[peak]/5 for peak in all_peaks})
        df3 = pd.concat([df3, pd.DataFrame([row3])], ignore_index=True)
    
    # 对于>600所有峰值不明显的，删除列。
    # 所以feature_num可能会比总的列数还大
    # cols_to_drop = []
    # for col in df2.columns:
    #     if col not in ['id', 'label', 'total_y', 'feature_num']:
    #         if df1[col].sum() < 10:
    #             cols_to_drop.append(col)
                
    # df1 = df1.drop(columns=cols_to_drop)
    # df2 = df2.drop(columns=cols_to_drop)
    # df3 = df3.drop(columns=cols_to_drop)

    save_path = make_path("corrected", "xlsx")
    df1.to_excel(save_path, index=False)
    print("corrected data successfully saved to", save_path)
    save_path = make_path("corrected_sum", "xlsx")
    print("corrected sum data successfully saved to", save_path)
    df2.to_excel(save_path, index=False)
    save_path = make_path("corrected_avg", "xlsx")
    df3.to_excel(save_path, index=False)
    print("corrected avg data successfully saved to", save_path)
    
if __name__ == "__main__":
    prepared_datas, reference_data,corrected_datas = peak_match()
    normed_datas = normalize(corrected_datas)
    # 保存数据
    save_data(reference_data, normed_datas)
    # 绘制相关系数热图
    # plot_correlation_heatmap(prepared_datas, corrected_datas)