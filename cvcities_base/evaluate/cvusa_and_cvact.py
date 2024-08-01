import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    """
   评估函数，用于评估模型在查询数据集上的性能

    参数：
        config (object): 配置对象
        model (object): 模型对象
        reference_dataloader (object): 参考数据的数据加载器对象
        query_dataloader (object): 查询数据的数据加载器对象
        ranks (list, optional): 需要计算的排名列表，默认为 [1, 5, 10]
        step_size (int, optional): 步长，默认为 1000
        cleanup (bool, optional): 是否进行内存清理，默认为 True

    返回值：
        r1 (dict): 包含查询数据在参考数据上的排名结果的字典
    """
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    print("计算分数：")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size,
                          ranks=ranks)

    # 清理并释放 GPU 内存
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1


def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    """
    计算相似度函数
    :param config: 配置参数
    :param model: 模型
    :param reference_dataloader: 参考数据加载器
    :param query_dataloader: 查询图像数据加载器
    :param ranks: 评估的排名列表，默认为[1, 5, 10]
    :param step_size: 步长，默认为1000
    :param cleanup: 是否清理内存，默认为True
    :return: 评估结果r1和near_dict
    """

    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    # reference_labels = reference_labels
    print("Compute Scores Train:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks)


    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)

    # 清理内存并释放GPU内存
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()

    return r1, near_dict


def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000,
                     ranks=[1, 5, 10]):
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)

    steps = Q // step_size + 1

    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()

    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i

    similarity = []

    for i in range(steps):
        start = step_size * i
        end = start + step_size

        sim_tmp = query_features[start:end] @ reference_features.T

        similarity.append(sim_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R // 100)

    results = np.zeros([len(topk)])

    bar = tqdm(range(Q))

    for i in bar:  # 遍历查询
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]

        # number of references with higher similiarity as gt  与 真实值 相似度较高的图像数
        higher_sim = similarity[i, :] > gt_sim

        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results / Q * 100.

    bar.close()

    # wait to close pbar
    time.sleep(0.1)

    string = []
    for i in range(len(topk) - 1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))

    string.append('Recall@top1: {:.4f}'.format(results[-1]))

    print(' - '.join(string))

    return results[0]


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):
    """
    计算最近邻
    参数:
    query_features: 查询样本的特征向量列表
    reference_features: 参考样本的特征向量列表
    query_labels: 查询样本的标签列表
    reference_labels: 参考样本的标签列表
    neighbour_range: 最近邻的范围，默认为64
    step_size: 步长，默认为1000
    返回:
    最近邻的字典
    """

    Q = len(query_features)

    steps = Q // step_size + 1

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_tmp = query_features[start:end] @ reference_features.T

        similarity.append(sim_tmp.cpu())

    # 将所有步数的相似度矩阵进行拼接，得到形状为 Q x R 的矩阵
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 1, dim=1)  # 获取相似度最高的邻近样本的索引

    topk_references = []  # 存储参考样本的标签列表

    for i in range(len(topk_ids)):  # 遍历每个查询样本
        topk_references.append(reference_labels[topk_ids[i, :]])  # 获取参考样本的标签

    topk_references = torch.stack(topk_references, dim=0)  # 将参考样本的标签列表转换为张量

    # 对于没有真实标签的样本，创建一个用于屏蔽的掩码
    mask = topk_references != query_labels.unsqueeze(1)

    topk_references = topk_references.cpu().numpy()  # 将参考样本的标签转换为 numpy 数组
    mask = mask.cpu().numpy()  # 将掩码转换为 numpy 数组

    # 创建一个字典，只存储相似度高于最低真实标签得分的样本的标签
    nearest_dict = dict()

    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]

        nearest_dict[query_labels[i].item()] = list(nearest)

    return nearest_dict  # 返回最近邻的字典


