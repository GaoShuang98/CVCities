import os
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader
from cvcities_base.dataset.cvusa import CVUSADatasetEval
from cvcities_base.transforms import get_transforms_val
from cvcities_base.evaluate.cvusa_and_cvact import evaluate, calc_sim
from cvcities_base.model import TimmModel
import shutil
from tqdm import tqdm
import random
from pathlib import Path
import sys
from cvcities_base.utils import Logger


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
    layer1 = -2
    use_cls = True
    norm_descs = True

    # Aggregator 聚合方法
    agg_arch = 'MixVPR'
    agg_config = {'in_channels': 768,
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
    save_top_k_img_path = r'D:\python_code\XXXXXXXXX'
    save_top_k_img_num = 5
    
    # Dataset
    data_folder = "D:/Datasets/CVUSA"
    
    # Checkpoint to start from
    checkpoint_start = r'D:\python_code\XXXXXXXXXXX'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 36
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#

    file_name, ext = os.path.splitext(os.path.basename(config.checkpoint_start))
    path = Path(config.checkpoint_start)
    save_path = path.parent.absolute()

    sys.stdout = Logger(os.path.join(save_path, 'cvusa_' + file_name + '_test_log.txt'))

    print("\nModel: {}".format(config.model))

    model = TimmModel(model_name=config.model,
                      pretrained=True,
                      img_size=config.img_size, backbone_arch=config.backbone_arch, agg_arch=config.agg_arch,
                      agg_config=config.agg_config, layer1=config.layer1)
                          
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
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
        
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
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

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVUSA", 30*"-"))

    if config.save_top_k_img:
        r1_test, near_dict = calc_sim(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        data_folder = config.data_folder
        save_folder = config.save_top_k_img_path
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(f'{save_folder}/queries'):
            os.makedirs(f'{save_folder}/queries')
        if not os.path.exists(f'{save_folder}/references'):
            os.makedirs(f'{save_folder}/references')
        for i, k in enumerate(tqdm(near_dict)):
            if i >= 8000:
                break
            v = near_dict[k]
            query_img_dir = f'{data_folder}/{query_dataset_test.idx2ground[k]}'
            print(query_img_dir)
            shutil.copy(query_img_dir, f"{save_folder}/queries/{query_dataset_test.idx2ground[k].split('/')[-1]}")
            for num in range(config.save_top_k_img_num):
                random_num = random.randint(0, 10000)
                if num == 0 and random_num < 9253:
                    reference_img_dir = f'{data_folder}/{reference_dataset_test.idx2sat[k]}'
                    save_sat_img_dir = f"{save_folder}/references/{query_dataset_test.idx2sat[k].split('/')[-1].split('.')[0]}@{str(num)}@{query_dataset_test.idx2ground[k].split('/')[-1].split('.')[0]}.jpg"
                    print(save_sat_img_dir)
                    shutil.copy(reference_img_dir, save_sat_img_dir)
                else:
                    reference_img_dir = f'{data_folder}/{reference_dataset_test.idx2sat[v[num]]}'
                    save_sat_img_dir = f"{save_folder}/references/{query_dataset_test.idx2sat[k].split('/')[-1].split('.')[0]}@{str(num)}@{reference_dataset_test.idx2sat[v[num]].split('/')[-1].split('.')[0]}.jpg"
                    print(save_sat_img_dir)
                    shutil.copy(reference_img_dir, save_sat_img_dir)

    else:
        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
