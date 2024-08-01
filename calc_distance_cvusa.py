import pandas as pd
from sklearn.metrics import DistanceMetric
import torch
import pickle

# 定义一个变量top-K值
TOP_K = 128

# 读取训练集数据并显示长度
df_train = pd.read_csv('D:/Datasets/CVUSA/splits/train-19zl.csv', header=None)
df_train = df_train.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
df_train["idx"] = df_train.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
train_sat_ids = df_train["idx"].values
print("Length Train Ids:", len(train_sat_ids))

# 读取所有位置数据并显示长度
df_gps = pd.read_csv('D:/Datasets/CVUSA/split_locations/all.csv', header=None)
df_gps = df_gps.rename(columns={0: "sat_lat", 1: "sat_long", 2: "ground_lat", 3: "ground_long", 4: "i_dont_know"})
gps_coords = {}
gps_coords_list = []
for idx in train_sat_ids :
    row = df_gps.iloc[idx-1]
    coordinates = (float(row["ground_lat"]), float(row["ground_long"]))
    gps_coords[idx] = coordinates
    gps_coords_list.append(coordinates)
print("Length of gps coords : " +str(len(gps_coords_list)))
print("Calculation...")
# 计算两个位置之间的距离矩阵
dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)
# 将距离矩阵转换为PyTorch张量，并填充对角线为最大值
dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())
# 获取前K个最小值
values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)
values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()
# 创建一个字典来存储每个训练图像的近邻
near_neighbors = dict()
for i, idx in enumerate(train_sat_ids):
    near_neighbors[idx] = train_sat_ids[ids_near_numpy[i]].tolist()
print("Saving...")
# 将近邻字典保存为Pickle文件
with open("D:/Datasets/CVUSA/gps_dict.pkl", "wb") as f:
    pickle.dump(near_neighbors, f)

