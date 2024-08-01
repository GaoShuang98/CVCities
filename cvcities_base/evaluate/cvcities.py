import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.nn.functional as F

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


def calc_sim_no_r1(config,
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

    print("\nExtract reference Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    print("\nExtract query Features:")
    query_features, query_labels = predict(config, model, query_dataloader)

    print("Compute Scores Train:")
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

    return near_dict

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

        del sim_tmp
        torch.cuda.empty_cache()

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

    del similarity
    torch.cuda.empty_cache()

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


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64,
                      step_size=1000):
    Q = len(query_features)
    steps = Q // step_size + 1
    nearest_dict = {}

    for step_i in range(steps):
        start = step_size * step_i
        end = min(start + step_size, Q)

        # 计算当前批次的相似度
        sim_batch = query_features[start:end] @ reference_features.T

        # 找出每个查询样本的top-k最相似的参考样本索引
        _, topk_ids_batch = torch.topk(sim_batch, k=neighbour_range + 1, dim=1)

        # 获取对应的参考样本标签
        topk_refs_batch = reference_labels[topk_ids_batch]

        # 创建掩码，排除与查询样本相同标签的参考样本
        mask = topk_refs_batch != query_labels[start:end].unsqueeze(1)

        # 分批构建nearest_dict
        for i in range(end - start):
            nearest = topk_refs_batch[i][mask[i]][:neighbour_range].tolist()
            nearest_dict[query_labels[start + i].item()] = nearest

        # 清理本批次使用的张量，释放内存
        del sim_batch, topk_ids_batch, topk_refs_batch, mask

    return nearest_dict


def calculate_scores_top5(config, model, reference_dataloader, query_dataloader, step_size=1000, topk=5):

    print("\nExtract reference Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    print("\nExtract query Features:")
    query_features, query_labels = predict(config, model, query_dataloader)

    Q = len(query_features)
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

        del sim_tmp
        torch.cuda.empty_cache()

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    # 归一化相似度矩阵以得到余弦相似度
    normalization = F.normalize(query_features, p=2, dim=1)
    similarity_normalized = F.normalize(similarity.clone().detach(), p=2, dim=1)

    # 计算每个查询的topk相似参考样本的索引
    topk_indices = []
    for i in range(Q):
        sorted_sim, sorted_indices = torch.topk(
            similarity_normalized[i].unsqueeze(0), topk, largest=True, sorted=True
        )
        topk_indices.append(sorted_indices[0].tolist())

    # 根据索引获取topk参考样本的标签
    topk_labels = [ref2index[idx] for indices in topk_indices for idx in indices]

    # 创建字典，键为query的label，值为top5 label列表
    result_dict = {}
    for i, query_label in enumerate(query_labels_np):
        result_dict[query_label] = [reference_labels_np[label] for label in topk_labels[i * topk:(i + 1) * topk]]

    return result_dict


