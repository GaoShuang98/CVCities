import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from cvcities_base.dataset.cvact import CVACTDatasetEval, CVACTDatasetTest
from cvcities_base.transforms import get_transforms_val
from cvcities_base.evaluate.cvusa_and_cvact import evaluate, calc_sim
from cvcities_base.model import TimmModel
import shutil
import scipy.io as sio
from pathlib import Path
import sys
from cvcities_base.utils import Logger
import numpy as np

@dataclass
class Configuration:

    # Model
    model = 'dinov2_vits14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vits14'
    # backbone_arch = ''
    pretrained = True

    layer1 = 2
    use_cls = True
    norm_descs = True

    # Aggregator 聚合方法
    agg_arch = 'MixVPR'  # CosPlace, NetVLAD, GeM
    agg_config = {'in_channels': 384,
                    'in_h': 32,
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
    batch_size: int = 180
    verbose: bool = True
    gpu_ids: tuple = (0, 1)
    normalize_features: bool = True
    neighbour_range = 64

    save_top_k_img = False
    save_top_k_img_path = r'//cvact/dinov2_vitb14_MixVPR/155158/top_k_img_2'
    save_top_k_img_num = 5

    # Dataset
    data_folder = "D:/Datasets/CVACT"
        
    # Checkpoint to start from
    checkpoint_start = r'D:\python_code\XXXXXXXXXXXXXXX'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 4
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration()

if __name__ == '__main__':


    file_name, ext = os.path.splitext(os.path.basename(config.checkpoint_start))
    path = Path(config.checkpoint_start)
    save_path = path.parent.absolute()

    sys.stdout = Logger(os.path.join(save_path, 'cvact_' + file_name + '_test_only_log.txt'))
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch, agg_config=config.agg_config, layer1=config.layer1)

    print(model)
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)
    
    # new_width = config.img_size * 2
    # new_hight = round((224 / 1232) * new_width)
    new_width = config.new_width
    new_hight = config.new_hight
    img_size_ground = (new_hight, new_width)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # Transforms                                                                  #
    #-----------------------------------------------------------------------------#

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )

    #-----------------------------------------------------------------------------#
    # Validation                                                                  #
    #-----------------------------------------------------------------------------#

    # # Reference Satellite Images
    # reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
    #                                          split="val",
    #                                          img_type="reference",
    #                                          transforms=sat_transforms_val,
    #                                          )
    #
    # reference_dataloader_val = DataLoader(reference_dataset_val,
    #                                       batch_size=config.batch_size,
    #                                       num_workers=config.num_workers,
    #                                       shuffle=False,
    #                                       pin_memory=True)
    #
    #
    #
    # # Query Ground Images Test
    # query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
    #                                      split="val",
    #                                      img_type="query",
    #                                      transforms=ground_transforms_val,
    #                                      )
    #
    # query_dataloader_val = DataLoader(query_dataset_val,
    #                                   batch_size=config.batch_size,
    #                                   num_workers=config.num_workers,
    #                                   shuffle=False,
    #                                   pin_memory=True)
    #
    #
    # print("Reference Images Val:", len(reference_dataset_val))
    # print("Query Images Val:", len(query_dataset_val))
    #
    # print("\n{}[{}]{}".format(30*"-", "CVACT_VAL", 30*"-"))
    #
    #
    #
    # if config.save_top_k_img:
    #     r1_test, near_dict = calc_sim(config=config,
    #                        model=model,
    #                        reference_dataloader=reference_dataloader_val,
    #                        query_dataloader=query_dataloader_val,
    #                        ranks=[1, 5, 10],
    #                        step_size=1000,
    #                        cleanup=True)
    #
    #     data_folder = config.data_folder
    #     save_folder = config.save_top_k_img_path
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)
    #     if not os.path.exists(f'{save_folder}/queries'):
    #         os.makedirs(f'{save_folder}/queries')
    #     if not os.path.exists(f'{save_folder}/references'):
    #         os.makedirs(f'{save_folder}/references')
    #     for k in near_dict:
    #         v = near_dict[k]
    #         query_img_dir = f'{data_folder}/ANU_data_test/streetview/{query_dataset_val.samples[k]}_grdView.jpg'
    #         # shutil.copy(query_img_dir, f'{save_folder}/queries/{query_dataset_val.samples[k]}_grdView.jpg')
    #         for num in range(config.save_top_k_img_num):
    #             reference_img_dir = f'{data_folder}/ANU_data_test/satview_polish/{reference_dataset_val.samples[v[num]]}_satView_polish.jpg'
    #             print(reference_img_dir + '--->' + f'{query_dataset_val.samples[k]}@satView@{str(num)}.jpg')
    #             # shutil.copy(reference_img_dir, f'{save_folder}/references/{query_dataset_val.samples[k]}@satView_{str(num)}.jpg')
    #
    # else:
    #     # pass
    #     r1_test = evaluate(config=config,
    #                        model=model,
    #                        reference_dataloader=reference_dataloader_val,
    #                        query_dataloader=query_dataloader_val,
    #                        ranks=[1, 5, 10],
    #                        step_size=1000,
    #                        cleanup=True)
    #-----------------------------------------------------------------------------#
    # Test                                                                        #
    #-----------------------------------------------------------------------------#
    
    # Reference Satellite Images Test
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
                                              img_type="reference",
                                              transforms=sat_transforms_val)
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)


    # Query Ground Images Test
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))          


    print("\n{}[{}]{}".format(30*"-", "CVACT_TEST", 30*"-"))

    if config.save_top_k_img:
        r1_test, near_dict = calc_sim(config=config,
                                      model=model,
                                      reference_dataloader=reference_dataloader_test,
                                      query_dataloader=query_dataloader_test,
                                      ranks=[1, 5, 10],
                                      step_size=1000,
                                      cleanup=True)

    else:
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
