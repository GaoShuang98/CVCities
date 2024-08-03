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
    评估函数，用于评估模型在参考数据集和查询数据集上的性能。

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
    print("\n提取特征：")
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
    :param query_dataloader: 查询数据加载器
    :param ranks: 评估的排名列表，默认为[1, 5, 10]
    :param step_size: 步长，默认为1000
    :param cleanup: 是否清理内存，默认为True
    :return: 评估结果r1和near_dict
    """
    print("\n提取特征:")
    reference_features, reference_labels = predict(config, model, reference_dataloader)
    query_features, query_labels = predict(config, model, query_dataloader)

    print("计算得分（训练集）:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size,
                          ranks=ranks)

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
    topk = copy.deepcopy(ranks)  # 存储要计算的topk值
    Q = len(query_features)  # 查询特征数量
    R = len(reference_features)  # 引用特征数量

    steps = Q // step_size + 1  # 计算分块步骤数

    query_labels_np = query_labels.cpu().numpy()  # 将查询标签转换为numpy数组
    reference_labels_np = reference_labels.cpu().numpy()  # 将引用标签转换为numpy数组

    ref2index = dict()  # 创建字典，用于存储引用标签和索引的映射关系
    for i, idx in enumerate(reference_labels_np):  # 遍历引用标签的索引和值
        ref2index[idx] = i  # 将引用标签和索引存储到字典中

    similarity = []  # 创建空列表，用于存储相似度矩阵

    for i in range(steps):  # 遍历分块步骤
        start = step_size * i  # 计算起始索引
        end = start + step_size  # 计算结束索引

        sim_tmp = query_features[start:end] @ reference_features.T  # 计算相似度矩阵

        similarity.append(sim_tmp.cpu())  # 将相似度矩阵添加到列表中

    similarity = torch.cat(similarity, dim=0)  # 将所有分块的相似度矩阵连接成一个矩阵，shape为(Q, R)

    topk.append(R // 100)  # 将引用数量除以100后补零添加到topk列表中

    results = np.zeros([len(topk)])  # 创建一个长度为topk列表长度的全零数组，用于存储结果

    bar = tqdm(range(Q))  # 创建进度条迭代器，迭代次数为查询数量

    for i in bar:  # 遍历查询
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]  # 获取查询的相似度值（与真实引用的）

        higher_sim = similarity[i, :] > gt_sim  # 获取与查询的相似度值高于真实引用的相似度矩阵

        ranking = higher_sim.sum()  # 统计高于真实引用的相似度值的数量
        for j, k in enumerate(topk):  # 遍历topk列表
            if ranking < k:  # 如果数量小于topk值
                results[j] += 1.  # 将结果数组对应位置的值加1

    results = results / Q * 100.  # 计算平均召回率

    bar.close()  # 关闭进度条

    time.sleep(0.1)  # 等待0.1秒

    string = []  # 创建空列表，用于存储结果字符串
    for i in range(len(topk) - 1):  # 遍历topk列表长度减1的范围
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))  # 将结果字符串添加到列表中

    string.append('Recall@top1: {:.4f}'.format(results[-1]))  # 将top1的召回率结果添加到列表中

    print(' - '.join(string))  # 打印结果字符串

    return results[0]  # 返回第一次查询的召回率结果


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64,
                      step_size=1000):
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
    Q = len(query_features)  # 查询样本的数量

    steps = Q // step_size + 1  # 计算计算最近邻的步骤数

    similarity = []  # 存储相似度的列表

    for i in range(steps):  # 对步骤进行循环
        start = step_size * i  # 计算起始位置

        end = start + step_size  # 计算结束位置

        sim_tmp = query_features[start:end] @ reference_features.T  # 计算查询样本与参考样本的相似度

        similarity.append(sim_tmp.cpu())  # 将相似度添加到相似度列表中

    # 将相似度列表中的所有相似度拼接成一个矩阵 Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 1, dim=1)  # 获取相似度最高的最近邻

    topk_references = []  # 存储参考样本的标签列表

    for i in range(len(topk_ids)):  # 对参考样本的标签列表进行循环
        topk_references.append(reference_labels[topk_ids[i, :]])  # 获取参考样本的标签

    topk_references = torch.stack(topk_references, dim=0)  # 将参考样本的标签列表转换为张量

    # 对于没有真实标签的样本，设置掩码为False
    mask = topk_references != query_labels.unsqueeze(1)

    topk_references = topk_references.cpu().numpy()  # 将参考样本的标签张量转换为numpy数组
    mask = mask.cpu().numpy()  # 将掩码张量转换为numpy数组

    # 创建一个字典，只存储相似度高于最低真实标签分数的最近邻
    nearest_dict = dict()

    for i in range(len(topk_references)):  # 对参考样本的标签列表进行循环
        nearest = topk_references[i][mask[i]][:neighbour_range]  # 获取最近邻标签

        nearest_dict[query_labels[i].item()] = list(nearest)  # 将最近邻标签添加到字典中

    return nearest_dict  # 返回最近邻字典


