import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from match import peak_match
from prepare import prepare, plot_patient_data

def TICnorm(corrected_datas):
    normed_datas = []
    normed_patient = []
    I = 1e6
    for patient in corrected_datas:
        normed_data = []
        for data in patient["datas"]:
            x, y = data
            # 计算TIC
            tic = np.sum(y)
            # 归一化
            y_normed = y / tic * I
            normed_data.append(np.array([x, y_normed]))
        normed_patient = {
            "name": patient["name"],
            "datas": normed_data
        }
        normed_datas.append(normed_patient)
    return normed_datas

def GBP(corrected_datas):
    normed_datas = TICnorm(corrected_datas)
    all_x = normed_datas[0]["datas"][0][0]
    len_x = len(all_x)
    num = len(normed_datas)
    five_means = []
    mean_of_five_means = []
    for i in range(len_x):  # 对于第i个荷质比
        means = []  # 第i个荷质比的五个均值
        for j in range(5):   # 计算每个板子的均值
            mean=0
            for patient in normed_datas:
                data = patient["datas"][j]
                x, y = data
                mean += y[i]
            mean /= num
            means.append(mean)
        five_means.append(means)
        mean_of_five_means.append(np.mean(means))
    profiled_datas = []
    profiled_data = []
    profiled_patient = []
    for patient in normed_datas:
        profiled_data = []
        for i in range(5):
            data = patient["datas"][i]
            x, y = data
            profiled_y = []
            for j in range(len_x):
                profiled_y.append(max(0,y[j] + mean_of_five_means[j] - five_means[j][i]))
            profiled_data.append(np.array([x, profiled_y]))
        profiled_patient = {
            "name": patient["name"],
            "datas": profiled_data
        }
        profiled_datas.append(profiled_patient)
    return profiled_datas

def normalize(corrected_datas):
    return GBP(corrected_datas)

if __name__ == "__main__":
    prepared_datas, reference_data,corrected_datas = peak_match()
    profiled_datas = GBP(corrected_datas)
    plot_patient_data(profiled_datas, get_all=True, num=0, xlim=(0, 1000), datas_labels=["A1_1", "A1_2", "A1_3", "A1_4", "A1_5"])
    