import numpy as np
from sklearn.metrics import DistanceMetric
from cvcities_base.dataset.cvact import CVACTDatasetTrain
import scipy.io as sio
import pickle
import torch

# 设置TOP_K值
TOP_K = 128

# 创建CVACTDatasetTrain对象，指定数据文件夹路径
dataset = CVACTDatasetTrain(data_folder="D:/Datasets/CVACT")

# 从文件中加载utm和panoIds数据
anuData = sio.loadmat('D:/Datasets/CVACT/ACT_data.mat')
utm = anuData["utm"]
ids = anuData['panoIds']

# 获取训练集中的样本id与对应的序号映射关系
idx2numidx = dataset.idx2numidx

# 创建训练集样本id集合
train_ids_set = set(dataset.train_ids)

# 创建训练集样本id列表和对应的序号列表
train_idsnum_list = []

# 创建utm坐标字典和utm坐标列表
utm_coords = dict()
utm_coords_list = []

# 遍历ids列表
for i, idx in enumerate(ids):
    # 将idx转换为字符串类型
    idx = str(idx)

    # 判断idx是否在训练集样本id集合中
    if idx in train_ids_set:
        # 获取utm坐标并添加到utm_coords字典中
        coordinates = (float(utm[i][0]), float(utm[i][1]))
        utm_coords[idx] = coordinates
        # 将idx和对应的序号添加到列表中
        utm_coords_list.append(coordinates)
        train_idsnum_list.append(idx2numidx[idx])

# 打印训练集样本id数量
print("Length Train Ids:", len(utm_coords_list))

# 将训练集样本id列表转换为numpy数组
train_idsnum_lookup = np.array(train_idsnum_list)

# 打印utm坐标数量
print("Length of gps coords: " + str(len(utm_coords_list)))
print("Calculation...")

# 使用欧氏距离计算距离矩阵
dist = DistanceMetric.get_metric("euclidean")
dm = dist.pairwise(utm_coords_list, utm_coords_list)
# 打印距离矩阵形状
print("Distance Matrix:", dm.shape)

# 将距离矩阵转换为torch张量，并将对角线填充为距离矩阵的最大值
dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

# 获取距离矩阵中前K个最小值的值和索引
values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

# 将值和索引转换为numpy数组
values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()

# 创建近邻字典
near_neighbors = dict()

# 遍历训练集样本id列表
for i, idnum in enumerate(train_idsnum_list):
    # 将近邻样本id添加到字典中
    near_neighbors[idnum] = train_idsnum_lookup[ids_near_numpy[i]].tolist()

# 保存近邻字典到文件中
print("Saving...")
with open("E:/datasets/CVACT/gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)
