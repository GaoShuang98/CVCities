import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from cvcities_base.dataset.cvcities import CVCITIESDatasetEval, CVCITIESDatasetTrain
from cvcities_base.cvcities_transforms import get_transforms_train, get_transforms_val
from cvcities_base.utils import setup_system, Logger
from cvcities_base.trainer import train
from cvcities_base.evaluate.cvcities import evaluate, calc_sim, calc_sim_no_r1
from cvcities_base.loss import InfoNCE

from cvcities_base.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    # model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    # model = 'convnext_base_384_in22ft1k'
    model = 'dinov2_vitb14_MixVPR'

    # backbone
    backbone_arch = 'dinov2_vitb14'

    pretrained = False  # False：patch embed 卷积模块 和 blocks 参数更新，True:patch embed 卷积模块 参数不更新 和 blocks 由layer1定义更新参数层数

    layer1 = -2
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
    # crop_size_ratio_min = 0.5
    # crop_size_ratio_max = 1.0
    crop_p = 1
    img_size: int = 448

    
    # Training 
    mixed_precision: bool = True
    seed = 1
    epochs: int = 40
    batch_size: int = 16    # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0, 1)   # GPU ids for training

    # Similarity Sampling
    custom_sampling: bool = True   # use custom sampling instead of random
    gps_sample: bool = True        # use gps sampling
    sim_sample: bool = True        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = r"D:\Datasets\CVCITIES\gps_dict_10_cities.pkl"   # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 72
    eval_every_n_epoch: int = 1        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    use_sgd = True
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = r'D:\Datasets\CVCITIES'
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "./cvcities"
    
    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    # checkpoint_start = r'D:\python_code\Sample4Geo-main\Sample4Geo-main\cvcities\dinov2_vitb14_MixVPR\2024-05-19_154851-mix1-part\weights_e2_52.6779.pth'
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 4
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False

    TRAIN_CITIES = [

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
        ######   sum 162954
        # 'tainan',  # 6711
        # 'taizhong',  # 6959
        # 'taoyuan',  # 7026
        # 'xinzhu',  # 6700
        # 'gaoxiong'  # 6971
        ]


    TEST_CITIES = [
        # 'taipei',  # 10842
        'singapore',  # 5564
        # 'london',  # 13617
        # 'seattle',  # 13654
        # 'rio',  # 13165
        # 'sydney',  # 3940
        #   sum 60782
        ]

#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':
    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%Y-%m-%d_%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    print("\nModel: {}".format(config.model))

    # 加载模型结构
    model = TimmModel(model_name=config.model,
                      pretrained=config.pretrained,
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
    new_width = config.img_size
    new_hight = config.img_size
    img_size_ground = (new_hight, new_width)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)
        del model_state_dict

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = model.to(config.device)
        print("Model: DataParallel")

    # Model to device
    # model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   # crop_size_ratio_min=config.crop_size_ratio_min,
                                                                   # crop_size_ratio_max=config.crop_size_ratio_max,
                                                                   crop_p=config.crop_p
                                                                   )
                                                                   
    # Train
    train_dataset = CVCITIESDatasetTrain(data_folder=config.data_folder,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      cities=config.TRAIN_CITIES
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )

    # # Reference Satellite Images
    # reference_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
    #                                           split="test",
    #                                           img_type="reference",
    #                                           transforms=sat_transforms_val,
    #                                           train_cities=config.TRAIN_CITIES,
    #                                           test_cities=config.TEST_CITIES
    #                                           )
    #
    # reference_dataloader_test = DataLoader(reference_dataset_test,
    #                                        batch_size=config.batch_size_eval,
    #                                        num_workers=config.num_workers,
    #                                        shuffle=False,
    #                                        pin_memory=True)
    #
    #
    # # Query Ground Images Test
    # query_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder ,
    #                                       split="test",
    #                                       img_type="query",
    #                                       transforms=ground_transforms_val,
    #                                       train_cities=config.TRAIN_CITIES,
    #                                       test_cities=config.TEST_CITIES
    #                                       )
    #
    # query_dataloader_test = DataLoader(query_dataset_test,
    #                                    batch_size=config.batch_size_eval,
    #                                    num_workers=config.num_workers,
    #                                    shuffle=False,
    #                                    pin_memory=True)
    #
    #
    # print("Reference Images Test:", len(reference_dataset_test))
    # print("Query Images Test:", len(query_dataset_test))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVCITIESDatasetEval(data_folder=config.data_folder,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               train_cities=config.TRAIN_CITIES,
                                               test_cities=config.TEST_CITIES
                                                  )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVCITIESDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   train_cities=config.TRAIN_CITIES,
                                                   test_cities=config.TEST_CITIES
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler 学习率调整/自适应优化/超参数调整等                                      #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))

        # Reference Satellite Images
        reference_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                     split="test",
                                                     img_type="reference",
                                                     transforms=sat_transforms_val,
                                                     train_cities=config.TRAIN_CITIES,
                                                     test_cities=config.TEST_CITIES
                                                     )

        reference_dataloader_test = DataLoader(reference_dataset_test,
                                               batch_size=config.batch_size_eval,
                                               num_workers=config.num_workers,
                                               shuffle=False,
                                               pin_memory=True)

        # Query Ground Images Test
        query_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                 split="test",
                                                 img_type="query",
                                                 transforms=ground_transforms_val,
                                                 train_cities=config.TRAIN_CITIES,
                                                 test_cities=config.TEST_CITIES
                                                 )

        query_dataloader_test = DataLoader(query_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)

        print("Reference Images Test:", len(reference_dataset_test))
        print("Query Images Test:", len(query_dataset_test))

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[{}/Epoch: {}]{}".format(30*"-",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),  epoch, 30*"-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if epoch != 0 or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            for test_city in config.TEST_CITIES:

                print(f'evaluating... test_city:------------{test_city}------------')

                # Reference Satellite Images
                reference_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder,
                                                          split="test",
                                                          img_type="reference",
                                                          transforms=sat_transforms_val,
                                                          train_cities=[test_city],
                                                          test_cities=[test_city]
                                                          )

                reference_dataloader_test = DataLoader(reference_dataset_test,
                                                       batch_size=config.batch_size_eval,
                                                       num_workers=config.num_workers,
                                                       shuffle=False,
                                                       pin_memory=True)

                # Query Ground Images Test
                query_dataset_test = CVCITIESDatasetEval(data_folder=config.data_folder ,
                                                      split="test",
                                                      img_type="query",
                                                      transforms=ground_transforms_val,
                                                      train_cities=[test_city],
                                                      test_cities=[test_city]
                                                      )

                query_dataloader_test = DataLoader(query_dataset_test,
                                                   batch_size=config.batch_size_eval,
                                                   num_workers=config.num_workers,
                                                   shuffle=False,
                                                   pin_memory=True)

                print("Reference Images Test:", len(reference_dataset_test))
                print("Query Images Test:", len(query_dataset_test))

                r1_test = evaluate(config=config,
                                   model=model,
                                   reference_dataloader=reference_dataloader_test,
                                   query_dataloader=query_dataloader_test,
                                   ranks=[1, 5, 10],
                                   step_size=1000,
                                   cleanup=True)

                del reference_dataloader_test
                del query_dataloader_test
                del reference_dataset_test
                del query_dataset_test

            print(f'eval config.TEST_CITIES end.')

            if r1_test > best_score:

                best_score = r1_test
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            if config.sim_sample:
                print('similarity sampling...')
                sim_dict = calc_sim_no_r1(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train,
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)

        if config.custom_sampling:
            print('custom sampling...')
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
