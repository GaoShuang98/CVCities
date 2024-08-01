import os
import torch
from dataclasses import dataclass
from tqdm import tqdm

from torch.utils.data import DataLoader
from cvcities_base.dataset.cvcities import CVCITIESDatasetEval
from cvcities_base.cvcities_transforms import get_transforms_val
from cvcities_base.evaluate.cvcities import evaluate, calc_sim
from cvcities_base.model import TimmModel
from pathlib import Path
import sys
from cvcities_base.utils import Logger
import shutil


# @dataclass
# class Configuration:  # eval classes for convnext_base_384_in22ft1k
#     # Model
#     model: str = 'convnext_base_384_in22ft1k'
#
#     # backbone
#     backbone_arch = 'convnext_base_384_in22ft1k'
#     pretrained = True  # False：patch embed 卷积模块 和 blocks 参数更新，True:patch embed 卷积模块 参数不更新 和 blocks 由layer1定义更新参数层数
#     # layers_to_freeze = 1
#     # layers_to_crop = []
#     layer1 = 8
#     use_cls = True
#     norm_descs = True
#
#     # Aggregator 聚合方法
#     agg_arch = 'MixVPR'  # CosPlace, NetVLAD, GeM
#     agg_config = {'in_channels': 768,
#                   'in_h': 32,  # 受输入图像尺寸的影响
#                   'in_w': 32,
#                   'out_channels': 1024,
#                   'mix_depth': 2,
#                   'mlp_ratio': 1,
#                   'out_rows': 4}
#     # Override model image size
#     # crop_size_ratio_min = 0.5
#     # crop_size_ratio_max = 1.0
#     crop_p = 1
#     img_size: int = 384
#
#     # Training
#     mixed_precision: bool = True
#     seed = 1
#     epochs: int = 40
#     batch_size: int = 32  # keep in mind real_batch_size = 2 * batch_size
#     verbose: bool = True
#     gpu_ids: tuple = (0, 1)  # GPU ids for training
#
#     # Similarity Sampling
#     custom_sampling: bool = True  # use custom sampling instead of random
#     gps_sample: bool = True  # use gps sampling
#     sim_sample: bool = True  # use similarity sampling
#     neighbour_select: int = 64  # max selection size from pool
#     neighbour_range: int = 128  # pool size for selection
#     gps_dict_path: str = r"D:\Datasets\CVCITIES\gps_dict_10_cities.pkl"  # path to pre-computed distances
#
#     # Eval
#     batch_size_eval: int = 72
#     eval_every_n_epoch: int = 4  # eval every n Epoch
#     normalize_features: bool = True
#
#     # Optimizer
#     clip_grad = 100.  # None | float
#     decay_exclue_bias: bool = False
#     grad_checkpointing: bool = False  # Gradient Checkpointing
#     use_sgd = False
#
#     # Loss
#     label_smoothing: float = 0.1
#
#     # Learning Rate
#     lr: float = 0.001  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
#     scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
#     warmup_epochs: int = 1
#     lr_end: float = 0.0001  # only for "polynomial"
#
#     # Dataset
#     data_folder = r'D:\Datasets\CVCITIES'
#
#     # Augment Images
#     prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
#     prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously
#
#     # Savepath for model checkpoints
#     model_path: str = "./cvcities"
#
#     # Eval before training
#     zero_shot: bool = False
#
#     # Checkpoint to start from
#     checkpoint_start = r'D:\python_code\XXX\weights_e36_68.4040.pth'
#
#     # set num_workers to 0 if on Windows
#     num_workers: int = 4
#
#     # train on GPU if available
#     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # for better performance
#     cudnn_benchmark: bool = True
#
#     # make cudnn deterministic
#     cudnn_deterministic: bool = False
#
#     TEST_CITIES = [
#
#         # test cities:
#         'taipei',  # 10842
#         'singapore',  # 5564
#         'london',  # 13617
#         'seattle',  # 13654
#         'rio',  # 13165
#         'sydney',  # 3940
#         #   sum 60782
#     ]
#
#     EVAL_TRAIN_CITIES = [
#         # train cities:
#         'barcelona',  # 18364
#         'captown',  # 10586
#         'losangeles',  # 18178
#         'maynila',  # 9971
#         'melbourne',  # 18247
#         'mexico',  # 19805
#         'newyork',  # 19996
#         'paris',  # 20214
#         'santiago',  # 16525
#         'tokyo',  # 11068
#     ]


@dataclass
class Configuration:
    # Model
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'
    # backbone_arch = ''
    pretrained = True
    # layers_to_freeze = 1
    # layers_to_crop = []
    layer1 = 7
    use_cls = True
    norm_descs = True

    # Aggregator 聚合方法
    agg_arch = 'MixVPR'  # CosPlace, NetVLAD, GeM
    agg_config = {'in_channels': 768,
                  'in_h': 32,  # 受输入图像尺寸的影响
                  'in_w': 32,
                  'out_channels': 1024,
                  'mix_depth': 2,
                  'mlp_ratio': 1,
                  'out_rows': 4}

    # Override model image size
    img_size: int = 448
    new_hight = 448
    new_width = 448

    # Evaluation
    batch_size: int = 40
    verbose: bool = True
    gpu_ids: tuple = (0, 1)
    normalize_features: bool = True
    neighbour_range = 64

    # save_top_k_img = False
    # save_top_k_img_path = r'none'
    # save_top_k_img_num = 5

    # Dataset
    data_folder = r'D:\Datasets\CVCITIES'

    # Checkpoint to start from
    checkpoint_start = r'D:\python_code\XXXXXXX'
    # set num_workers to 0 if on Windows
    num_workers: int = 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    TEST_CITIES = [

        # test cities:
        'taipei',  # 10842
        'singapore',  # 5564
        'london',  # 13617
        'seattle',  # 13654
        'rio',  # 13165
        'sydney',  # 3940
        #   sum 60782
    ]

    EVAL_TRAIN_CITIES = [
        # train cities:
        'barcelona',  # 18364
        'captown',  # 10586
        'losangeles',  # 18178
        'maynila',  # 9971
        'melbourne',  # 18247
        'mexico',  # 19805
        'newyork',  # 19996
        'paris',  # 20214
        'santiago',  # 16525
        'tokyo',  # 11068
    ]


# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#

config = Configuration()

if __name__ == '__main__':

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    file_name, ext = os.path.splitext(os.path.basename(config.checkpoint_start))
    path = Path(config.checkpoint_start)
    save_path = path.parent.absolute()

    sys.stdout = Logger(os.path.join(save_path, 'cvcities_' + file_name + '_test_eval_log.txt'))

    print("\nModel: {}".format(config.model))

    model = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1)

    print(model)
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size

    image_size_sat = (img_size, img_size)

    # new_width = config.img_size * 2
    # new_hight = config.img_size
    new_width = config.new_width
    new_hight = config.new_hight
    img_size_ground = (new_hight, new_width)

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = model.to(config.device)
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        print("Model: DataParallel")

    # Model to device   
    # model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    for city in config.TEST_CITIES:
        # Reference Satellite Images
        reference_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                     split="test",
                                                     img_type="reference",
                                                     transforms=sat_transforms_val,
                                                     train_cities=[city],
                                                     test_cities=[city],
                                                     )

        reference_dataloader_test = DataLoader(reference_dataset_test,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

        # Query pano Images Test
        query_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                 split="test",
                                                 img_type="query",
                                                 transforms=ground_transforms_val,
                                                 train_cities=[city],
                                                 test_cities=[city],
                                                 )

        query_dataloader_test = DataLoader(query_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        # -----------------------------------------------------------------------------#
        # Evaluate                                                                    #
        # -----------------------------------------------------------------------------#

        print("\n{}[{}]{}".format(30 * "-", f"City name:{city}", 30 * "-"))
        print("Reference Images Test:", len(reference_dataset_test))
        print("Query Images Test:", len(query_dataset_test))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)


    if len(config.EVAL_TRAIN_CITIES) > 0:
        # Reference Satellite Images
        reference_dataset_eval_train = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                     split="test",
                                                     img_type="reference",
                                                     transforms=sat_transforms_val,
                                                     train_cities=config.EVAL_TRAIN_CITIES,
                                                     test_cities=config.EVAL_TRAIN_CITIES,
                                                     )

        reference_dataloader_eval_train = DataLoader(reference_dataset_eval_train,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

        # Query pano Images Test
        query_dataset_eval_train = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                 split="test",
                                                 img_type="query",
                                                 transforms=ground_transforms_val,
                                                 train_cities=config.EVAL_TRAIN_CITIES,
                                                 test_cities=config.EVAL_TRAIN_CITIES,
                                                 )

        query_dataloader_eval_train = DataLoader(query_dataset_eval_train,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        # -----------------------------------------------------------------------------#
        # Evaluate                                                                    #
        # -----------------------------------------------------------------------------#

        print("\n{}[{}]{}".format(30 * "-", f" Total city number :{len(config.EVAL_TRAIN_CITIES)}", 30 * "-"))
        print("Reference Images Test:", len(reference_dataset_eval_train))
        print("Query Images Test:", len(query_dataset_eval_train))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_eval_train,
                           query_dataloader=query_dataloader_eval_train,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
