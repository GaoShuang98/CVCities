import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time


class CVCITIESDatasetTrain(Dataset):

    def _read_cities_csv(self, cities, data_folder):
        all_df = pd.DataFrame()
        for city in cities:
            df = pd.read_csv(f'{data_folder}/{city}/img_info.csv', header=None)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        return all_df

    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 cities=['taipei', 'maynila']
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.cities = cities
        self.df = self._read_cities_csv(self.cities, data_folder)
        self.df = self.df.rename(columns={0:'name', 1:'longitude', 2:'latitude', 3:'city', 4:'sat_dir', 5:'pano_dir'})


        self.idx2sat = dict(zip(self.df.index, self.df.sat_dir))
        self.idx2pano = dict(zip(self.df.index, self.df.pano_dir))
        self.pairs = list(zip(self.df.index, self.df.sat_dir, self.df.pano_dir))

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)


    def __getitem__(self, index):
        """
             获取数据集中一个样本的查询图像、参考图像和标签
             Args:
                 index (int): 数据集中的样本索引
             Returns:
                 tuple: 包含查询图像、参考图像和标签的元组
             """
        # 从idx2pair字典中获取样本的索引、卫星图像路径和地面图像路径
        try:
            idx, sat_dir, pano_dir = self.idx2pair[self.samples[index]]
        except:
            print('stop')
        # idx, sat_dir, pano_dir = row.name, row['sat_dir'], row['pano_dir']

        # load query -> ground image 加载查询->地面图像
        query_img = cv2.imread(f'{self.data_folder}/{pano_dir}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image 加载参考-> 卫星图像
        reference_img = cv2.imread(f'{self.data_folder}/{sat_dir}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Flip simultaneously query and reference 同时翻转查询和引用
        if np.random.random() < self.prob_flip:  # 如果随机数小于prob_flip，则同时翻转查询图像和参考图像
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

            # image transforms 图像转换
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously query and reference 同时旋转查询t图像和引用图像
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            # rotate sat img 90 or 180 or 270 旋转卫星图像90度、180度或270度
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view 如果旋转卫星图像，则使用滚动作为地面视图
            c, h, w = query_img.shape
            shifts = - w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        '''
        custom shuffle function for unique class_id sampling in batch 自定义用于批次中唯一类ID采样的洗牌函数
        '''

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)  # 将训练ID复制到idx_pool中
        neighbour_split = neighbour_select // 2  # 计算邻居选择的一半

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)  # 随机洗牌idx_pool

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:   # 如果idx不在当前批次中且不在当前epoch中且当前批次中的idx数量小于洗牌批次大小

                    idx_batch.add(idx)   # 将idx添加到当前批次中
                    current_batch.append(idx)  # 将idx添加到当前批次的idx列表中
                    idx_epoch.add(idx)  # 将idx添加到当前epoch中
                    break_counter = 0   # 重置计数器

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:   # 如果sim_dict不为空且当前批次中的idx数量小于洗牌批次大小

                        near_similarity = similarity_pool[idx][:neighbour_range]  # 获取idx在similarity_pool中的邻居范围内的相似性

                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])  # 获取邻居选择的一半

                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])   # 获取邻居选择的一半之外的邻居

                        random.shuffle(far_neighbours)   # 随机洗牌far_neighbours

                        far_neighbours = far_neighbours[:neighbour_split]   # 获取邻居选择的一半

                        near_similarity_select = near_neighbours + far_neighbours   # 将近邻和远邻合并

                        for idx_near in near_similarity_select:   # 遍历合并后的近邻和远邻

                            # check for space in batch  检查是否有空间在批次中
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch 检查 IDX 是否尚未处于批处理或 epoch 状态
                            if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                                idx_batch.add(idx_near)  # 将idx_near添加到当前批次中
                                current_batch.append(idx_near)   # 将idx_near添加到当前批次的idx列表中
                                idx_epoch.add(idx_near)  # 将idx_near添加到当前epoch中
                                similarity_pool[idx].remove(idx_near)  # 从similarity_pool中移除idx_near
                                break_counter = 0  # 重置计数器

                else:  # 如果idx不适合在批次中且尚未处于批处理或epoch状态
                    # if idx fits not in batch and is not already used in epoch -> back to pool 如果idx不在当前批次中且不在当前epoch中，则将其返回到idx_pool中
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)  # 将idx添加到idx_pool中

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:   # 如果当前批次中的idx数量大于等于洗牌批次大小
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class CVCITIESDatasetEval(Dataset):

    def _read_cities_csv(self, cities, data_folder):
        all_df = pd.DataFrame()
        for city in cities:
            df = pd.read_csv(f'{data_folder}/{city}/img_info.csv', header=None)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        return all_df

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 train_cities= ['taipei', 'maynila'],
                 test_cities= ['taipei', 'maynila'],
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.train_cities = train_cities
        self.test_cities = test_cities

        if split == 'train':
            self.df = self._read_cities_csv(self.train_cities, self.data_folder)
        else:  # test
            self.df = self._read_cities_csv(self.test_cities, self.data_folder)

        self.df = self.df.rename(columns={0:'name',  1:'longitude', 2:'latitude', 3:'city', 4:'sat_dir', 5:'pano_dir'})

        self.idx2sat = dict(zip(self.df.index, self.df.sat_dir))
        self.idx2pano = dict(zip(self.df.index, self.df.pano_dir))

        if self.img_type == "reference":
            self.images = self.df.sat_dir.values
            self.label = self.df.index.tolist()

        elif self.img_type == "query":
            self.images = self.df.pano_dir.values
            self.label = self.df.index.tolist()
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")



    def __getitem__(self, index):

        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    s = CVCITIESDatasetTrain(data_folder="E:/datasets/CVCITIES_raw")
    # s = CVCITIESDatasetTrain(data_folder="I:/CVCities",)
    a = s[1]
    print(a)
    s.__getitem__(2)




