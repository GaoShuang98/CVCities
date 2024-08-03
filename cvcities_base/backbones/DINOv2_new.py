# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 20,  layer2=25, layer3=30, facet1: _DINO_FACETS = "value", facet2='value', use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()  # 将大写转化为小写

        self.layer3 = layer3
        # self.layer2 = layer2  # 要导出特征对应的层
        self.layer1 = layer1

        # self.facet1 = facet1  # 定于要取的 qkvt（query key value token）类型
        # self.facet2 = facet2


        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name

        # self._hook_out1 = None
        # self._hook_out2 = None
        self._hook_out3 = None

        print('loading DINOv2 model...')
        self.dino_model = torch.hub.load(r'D:\python_code\MixVPR(hgs)\models\backbones\facebookresearch_dinov2_main\dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
        print(self.dino_model)
        # if self.facet1 == "token":
        #     self.fh_handle1 = self.dino_model.blocks[self.layer1].register_forward_hook(self._generate_forward_hook1())
        # else:
        #     self.fh_handle1 = self.dino_model.blocks[self.layer1].attn.qkv.register_forward_hook(self._generate_forward_hook1())
        #
        # if self.facet2 == "token":
        #     self.fh_handle2 = self.dino_model.blocks[self.layer2].register_forward_hook(self._generate_forward_hook2())
        # else:
        #     self.fh_handle2 = self.dino_model.blocks[self.layer2].attn.qkv.register_forward_hook(self._generate_forward_hook2())
        # self.fh_handle2 = self.dino_model.blocks[self.layer2].mlp.w3.register_forward_hook(self._generate_forward_hook2())

        # self.fh_handle1 = self.dino_model.blocks[self.layer1].attn.proj.register_forward_hook(self._generate_forward_hook1())
        # self.fh_handle2 = self.dino_model.blocks[self.layer2].attn.proj.register_forward_hook(self._generate_forward_hook2())
        self.fh_handle3 = self.dino_model.blocks[self.layer3].attn.register_forward_hook(self._generate_forward_hook3())
        self.dino_model = self.dino_model.to(self.device)

        if pretrained:
        #     self.dino_model.patch_embed.requires_grad_(False)
        #     self.dino_model.blocks.requires_grad_(False)
        #     # self.dino_model.norm.requires_grad_(False)
        #     # self.dino_model.head.requires_grad_(True)
        #     self.dino_model.blocks[self.layer1-1].requires_grad_(True)

            # for i in range(0, self.layer1):
            for i in range(0, self.layer3):
                self.dino_model.blocks[i].requires_grad_(False)

            for i in range(self.layer3 + 1, 40):
                self.dino_model.blocks[i] = nn.Sequential()

        #     self.dino_model.blocks[self.layer1].attn = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].ls1 = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].drop_path = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].norm2 = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].mlp = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].ls2 = nn.Sequential()
        #     self.dino_model.blocks[self.layer1].drop_path2 = nn.Sequential()
        #     self.dino_model.norm = nn.Sequential()
        #     self.dino_model.head = nn.Sequential()

        self.dino_model.norm = nn.Sequential()
        self.dino_model.head = nn.Sequential()


    def _generate_forward_hook1(self):
        def _forward_hook(module, inputs, output1):
            self._hook_out1 = output1
        return _forward_hook

    def _generate_forward_hook2(self):
        def _forward_hook(module, inputs, output2):
            self._hook_out2 = output2
        return _forward_hook

    def _generate_forward_hook3(self):
        def _forward_hook(module, inputs, output3):
            self._hook_out3 = output3
        return _forward_hook


    def forward(self, x):

        # x = self.dino_model(x)
        # x = self._hook_out1[:, 1:, ...]


        x = self.dino_model(x)  # 先过一遍模型，然后从hook提取对应张量

        if self.use_cls:
            # x1 = self._hook_out1
            # x2 = self._hook_out2
            # x3 = self._hook_out3
            x = self._hook_out3
            # x = torch.cat([x1, x3], dim=1)
            # x = torch.cat([x1, x2, x3], dim=1)

        else:
            # x1 = self._hook_out1[:, 1:, ...]
            # x2 = self._hook_out2[:, 1:, ...]
            # x3 = self._hook_out3[:, 1:, ...]
            x = self._hook_out3[:, 1:, ...]
            #
            # if self.facet1 in ["query", "key", "value"]:
            #     d_len = x1.shape[2] // 3
            #     if self.facet1 == "query":
            #         x1 = x1[:, :, :d_len]
            #     elif self.facet1 == "key":
            #         x1 = x1[:, :, d_len:2 * d_len]
            #     else:
            #         x1 = x1[:, :, 2 * d_len:]
            #
            # if self.facet2 in ["query", "key", "value"]:
            #     d_len = x2.shape[2] // 3
            #     if self.facet2 == "query":
            #         x2 = x2[:, :, :d_len]
            #     elif self.facet2 == "key":
            #         x2 = x2[:, :, d_len:2 * d_len]
            #     else:
            #         x2 = x2[:, :, 2 * d_len:]

            # x = torch.cat([x1, x3], dim=1)
            # x = torch.cat([x1, x2, x3], dim=1)

        # if self.norm_descs:
        #     x = F.normalize(x, dim=-1)  # 对张量进行归一化操作

        # self._hook_out1 = None
        # self._hook_out2 = None
        self._hook_out3 = None

        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 3, 224, 224)
    model = DinoV2_(model_name='dinov2_vitg14', layer1=39, layer2=28, layer3=30,  facet1="value", facet2="value", use_cls=False, norm_descs=True, device="cuda", pretrained=True)

    print(model)
    print('-' * 70)
    summary(model, (3, 224, 224), 2, 'cuda')
    print('-' * 70)

    r = model(x.to('cuda'))
    print_nb_params(model)

    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')


if __name__ == '__main__':
    main()


