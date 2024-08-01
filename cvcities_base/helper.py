import numpy as np
from cvcities_base import aggregators
from cvcities_base import backbones
from torch import nn


def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=3,
                 layers_to_crop=[],
                 layer1=20,
                 use_cls=False,
                 norm_descs=True):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where 
                                         we sometimes need to crop the last 
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        print(f"backbone_arch:{backbone_arch}")
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)


    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained, 
                                          layers_to_freeze=layers_to_freeze)
            
    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k', 
                              pretrained=pretrained, 
                              layers_to_freeze=layers_to_freeze)
    elif 'emo' in backbone_arch.lower():
        return backbones.EMO_new(model_name=backbone_arch, layer_to_freeze=layers_to_freeze, layers_to_crop=layers_to_crop)

    # elif 'dino' in backbone_arch.lower():
    #     return backbones.DinoV2(model_name=backbone_arch, layer=layer, facet=facet)

    # elif 'dino' in backbone_arch.lower():
    #     return backbones.DinoV2_(model_name=backbone_arch, layer1=layer1, facet1=facet1, layer2=layer2, layer3=layer3, facet2=facet2, use_cls=use_cls, norm_descs=norm_descs)

    elif 'dino' in backbone_arch.lower():
        return backbones.DinoV2_self(model_name=backbone_arch, layer1=layer1, use_cls=use_cls, norm_descs=norm_descs)
def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        # assert 'in_h' in agg_config
        # assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)

    elif 'netvlad' in agg_arch.lower():
        return aggregators.NetVLAD(**agg_config)

    elif 'none' in agg_arch.lower():  # 原始网络结构（全连接层结构）
        return nn.Identity()