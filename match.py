import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from getdata import get_all_patients_datas, get_datas
from prepare import prepare, plot_patient_data
from scipy.interpolate import CubicSpline

def get_reference_data(prepared_datas):
    """
    获取参考数据
    返回一个字典，包含患者编号和数据
    """
    # 每一个data的峰值信息
    peaks_info=[]

    for patient_data in prepared_datas:
        datas = patient_data["datas"]
        for data in datas:
            x, y = data
            peak_info = {
                "name": patient_data["name"],
                "num": np.sum(y>0),
                "data": data
            }
            peaks_info.append(peak_info)  
    # 找到peaks_info_num的中位数
    peaks_info = sorted(peaks_info, key=lambda x: x["num"])
    median_index = len(peaks_info) // 2
    median_peak_info = peaks_info[median_index]
    # 输出选择了哪个患者的数据
    print("Median Peak Info:", median_peak_info)
    reference_data = {
        "name": "reference",
        "datas": [median_peak_info["data"]]# 和其他患者的数据格式一致，其他有五组，所以这里要改成list，方便画图
    }
    return reference_data

def roll_with_padding(arr, shift):
    """
    对数组进行移位操作，超出或不足的部分用零填充
    :param arr: 输入数组
    :param shift: 移位量，正数表示右移，负数表示左移
    :return: 移位后的数组
    """
    result = np.zeros_like(arr)  # 创建与输入数组形状相同的零数组
    if shift > 0:
        # 右移：前 shift 个位置填充零
        result[shift:] = arr[:-shift]
    elif shift < 0:
        # 左移：后 abs(shift) 个位置填充零
        result[:shift] = arr[-shift:]
    else:
        # 不移位
        result = arr
    return result

def peak_correct(ref_data, prepared_datas):
    """
    整体思路：按照参考谱图的每个峰值，尽量用测试谱图去凑。如果可以凑上，就把他平移到参考谱图的位置。
    之前尝试过的算法：直接整体平移。发现可以有效的改良相关性，但是会有很多小的峰值错位，在Excel里大概看到是1~2列的错位。因此改用了一个一个峰值去凑。
    """
    # 获取参考谱图数据
    ref_x, ref_y = ref_data["datas"][0]
    corrected_datas=[]
    peaks = np.where(ref_y > 0)[0]  # 获取参考谱图中峰的位置
    # 设置最大位移量。目测2~3已经足够。太大了单峰区域容易错配。
    max_shift = 4
    # 2*window是用来算相关系数的窗口大小，评估平移后峰附近相似性如何。
    window = 10
    for patient_data in prepared_datas:
        name = patient_data["name"]
        datas = patient_data["datas"]
        corrected_data = []
        print("正在校正患者数据：", name)
        for data in datas:
            x, y = data 
            new_y = np.zeros_like(y)
            for peak in peaks:
                if(peak-window<0 or peak+window>len(ref_y)):
                    continue
                max_correlation = 0
                best_shift = 0
                for shift in range(-max_shift, max_shift + 1):
                    # 将原始数据进行平移，shift为正表示右移，负表示左移
                    shifted_y = roll_with_padding(y, shift)
                    # 计算当前位移下数据的小窗口内的皮尔逊相关系数
                    corr = np.corrcoef(ref_y[peak-window:peak+window], shifted_y[peak-window:peak+window])[0, 1]
                    # 处理NaN值（可能会有）
                    if(np.isnan(corr)):
                        corr = 0
                    # 记录最大相关系数和对应的位移
                    if corr >= max_correlation:
                        max_correlation = corr
                        best_shift = shift
                if(np.abs((best_shift)==max_shift) & (max_correlation<0.8)):
                    # 如果到头了并且相关性不高，就不平移了
                    # 一般来说，相关性会在0.8以上
                    best_shift = 0
                # 注意这里下标是减的。
                new_y[peak] = y[peak-best_shift]
            corrected_data.append(np.array([x, new_y]))
        corrected_patient = {
            "name": name,
            "datas": corrected_data
        }
        corrected_datas.append(corrected_patient)
    return corrected_datas


def peak_match():
    prepared_datas = prepare()
    # 这里sorted是因为get_reference里面会对prepared_datas进行排序，这会更改数组外的顺序，影响计算。
    prepared_datas_sorted = prepared_datas.copy()
    reference_data = get_reference_data(prepared_datas_sorted)
    corrected_datas = peak_correct(reference_data, prepared_datas)
    return prepared_datas, reference_data, corrected_datas

if __name__ == "__main__":
    peak_match()

