import matplotlib.pyplot as plt
import os
import numpy as np

def make_file_path(name, get_all=False, num=0):
    """
    构造文件路径
    name:患者编号，如A1, B2, C3, ...
    get_all:是否返回所有路径（五个）, 默认为 False,选择true时 num 参数无效
    num:只返回该患者的一个路径，1~5.
    该函数不做数据是否存在的检查。
    """
    current_path = os.getcwd()
    file_paths = []
    if get_all:
        for i in range(1, 6):
            sample_name = "samples_0_" + name + "_" + str(i) + ".txt"
            file_paths.append(os.path.join(current_path, "txt", sample_name))
    else:
        sample_name = "samples_0_" + name + "_" + str(num) + ".txt"
        file_paths.append(os.path.join(current_path, "txt", sample_name))
    return file_paths

def get_datas(name, get_all=False, num=0):
    """
    name:患者编号，如A1, B2, C3, ...
    get_all:是否返回所有数据，默认为 False,选择true时 num 参数无效
    num:返回该患者的某一个数据，1~5.
    """
    file_paths = make_file_path(name, get_all, num)
    datas = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            x, y = [], []  # Reset x and y for each file
            with open(file_path, 'r') as file:
                for line in file:
                    # 按空格分隔每行数据
                    parts = line.strip().split()
                    if len(parts) == 2:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
            data = np.array([x, y])
            datas.append(data)
        else:
            return False
    return {"name":name,"datas":datas}

def get_all_patients_datas():
    """
    获取所有患者的数据
    返回一个列表，列表中的每个元素是一个字典，字典包含三个键：col, index, datas。
    col: 列名
    index: 行号
    """
    cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    # cols = ["P"]
    all_patients_datas = []  # 使用列表存储每个患者的数据
    """
    all_patients_datas 是一个列表，元素类型为：
    patient = {
        "name":患者编号,如A1, B2, C3, ...
        "datas":[
            data1,data2,data3,data4,data5
            #注：data1 = np.array([x1, y1])
        ]
    }
    """
    for col in cols:
        for i in range(1, 27):
            datas = get_datas(col + str(i), get_all=True)
            if datas:
                # 将每个患者的数据存储为字典
                all_patients_datas.append(datas)
            else:
                print(f"文件 {col + str(i)} 不存在，跳过该患者。")
                break
    return all_patients_datas

if __name__ == "__main__":
    gapd=get_all_patients_datas()