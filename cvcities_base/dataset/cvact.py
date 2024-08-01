import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob

class CVACTDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # 用于查询图像的变换
        self.transforms_reference = transforms_reference   # 用于参考图像的变换

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]

        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():

            idx = str(idx)

            grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'

            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1

        print("IDs在训练图像中未找到:", self.idx_ignor)

        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)


    def __getitem__(self, index):

        idnum = self.samples[index]

        idx = self.numidx2idx[idnum]

        # 加载查询图像 -> 地面图像
        query_img = cv2.imread(f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # 加载参考图像 -> 卫星图像
        reference_img = cv2.imread(f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)


        # 同时翻转查询和参考图像
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # 图像变换
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # 同时旋转查询和参考图像
        if np.random.random() < self.prob_rotate:

            r = np.random.choice([1,2,3])

            # 将卫星图像旋转90度或180度或270度
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # 对地面视图使用roll操作如果旋转了卫星视图
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)


        label = torch.tensor(idnum, dtype=torch.long)

        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)



    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

            '''
            自定义的洗牌函数，用于在批次中采样唯一类ID
            '''

            print("\nShuffle Dataset:")

            idx_pool = copy.deepcopy(self.train_idsnum)

            neighbour_split = neighbour_select // 2

            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)

            # 洗牌pairs的顺序
            random.shuffle(idx_pool)

            # 查找是否在当前epoch中已使用
            idx_epoch = set()
            idx_batch = set()

            # 污桶
            batches = []
            current_batch = []

            # 计数器
            break_counter = 0

            # 进度条
            pbar = tqdm()

            while True:

                pbar.update()

                if len(idx_pool) > 0:
                    idx = idx_pool.pop(0)


                    if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:

                        idx_batch.add(idx)
                        current_batch.append(idx)
                        idx_epoch.add(idx)
                        break_counter = 0

                        # 检查附近卫星视图是否在范围内
                        if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:

                            near_similarity = similarity_pool[idx][:neighbour_range]

                            near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])

                            far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])

                            random.shuffle(far_neighbours)

                            far_neighbours = far_neighbours[:neighbour_split]

                            near_similarity_select = near_neighbours + far_neighbours

                            for idx_near in near_similarity_select:

                                # 检查批次中是否有空间
                                if len(current_batch) >= self.shuffle_batch_size:
                                    break

                                # 检查idx是否在批次和当前epoch中已经使用，并且不在忽略列表（丢失图像）中
                                if idx_near not in idx_batch and idx_near not in idx_epoch:

                                    idx_batch.add(idx_near)
                                    current_batch.append(idx_near)
                                    idx_epoch.add(idx_near)
                                    similarity_pool[idx].remove(idx_near)
                                    break_counter = 0

                    else:
                        # 如果idx无法放入批次中，并且尚未在epoch中使用 -> 回到idx_pool中
                        if idx not in idx_epoch:
                            idx_pool.append(idx)

                        break_counter += 1

                    if break_counter >= 1024:
                        break

                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # 空的current_batch桶添加到batches中
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []

            pbar.close()

            # 等待再关闭进度条
            time.sleep(0.3)

            self.samples = batches
            print("idx_pool:", len(idx_pool))
            print("原始长度: {} - 随机洗牌后的长度: {}".format(len(self.train_ids), len(self.samples)))
            print("断点计数器:", break_counter)
            print("pairs遗漏批次以避免产生噪声:", len(self.train_ids) - len(self.samples))
            print("第一个元素ID: {} - 最后一个元素ID: {}".format(self.samples[0], self.samples[-1]))

            
       
class CVACTDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')
        
        ids = anuData['panoIds']
        
        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")  
            
        if img_type != 'query' and img_type != 'reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")


        ids = ids[anuData[f'{split}Set'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
   
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i+=1
        
        # print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list


    def __getitem__(self, index):
        
        idx = self.samples[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)

            
class CVACTDatasetTest(Dataset):
    
    def __init__(self,
                 data_folder,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms = transforms
        
        files_sat = glob(f'{self.data_folder}/ANU_data_test/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'{self.data_folder}/ANU_data_test/streetview/*_grdView.jpg')
        
        sat_ids = []
        for path in files_sat:
            idx = os.path.basename(path).removesuffix('_satView_polish.jpg')
            sat_ids.append(idx)
        
        ground_ids = []
        for path in files_ground:
            idx = os.path.basename(path).removesuffix('_grdView.jpg')
            ground_ids.append(idx)  

        # only use intersection of sat and ground ids   
        test_ids = set(sat_ids).intersection(set(ground_ids))
        
        self.test_ids = list(test_ids)
        self.test_ids.sort()
        
        self.idx2num_idx = dict()
        
        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i


    def __getitem__(self, index):
        
        idx = self.test_ids[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/ANU_data_test/satview_polish/{idx}_satView_polish.jpg'
        else:
            path = f'{self.data_folder}/ANU_data_test/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.test_ids)




