import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_vigor(train_config, model, dataloader):
    model.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img)

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

# def predict(train_config, model, dataloader):
#     model.eval()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     img_features_list = []
#
#     ids_list = []
#     with torch.no_grad():
#
#         for img, ids in bar:
#
#             ids_list.append(ids)
#
#             with autocast():
#
#                 img = img.to(train_config.device)
#                 img_feature = model(img)
#
#                 # normalize is calculated in fp32
#                 if train_config.normalize_features:
#                     img_feature = F.normalize(img_feature, dim=-1)
#
#             # save features in fp32 for sim calculation
#             img_features_list.append(img_feature.to(torch.float32))
#
#         # keep Features on GPU
#         img_features = torch.cat(img_features_list, dim=0)
#         ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
#
#     if train_config.verbose:
#         bar.close()
#
#     return img_features, ids_list