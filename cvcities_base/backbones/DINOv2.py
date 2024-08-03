# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]


class DinoV2(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer: int = 31, facet: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda", pretrained=True) -> None:
        """
        Parameters:
        - model_name:   The DINO-v2 model to use
        - layer:        The layer to extract features from
        - facet:        "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
        - use_cls:      If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
        - norm_descs:   If True, the descriptors are normalized
        - device:       PyTorch device to use
        - pretrained:   If True, load pretrained weights
        """
        super().__init__()
        if pretrained:
            pass

        self.vit_type: str = model_name
        print('loading DINOv2 model...')
        self.dino_model: nn.Module = torch.hub.load(r'D:\Python_code\MixVPR(hgs)\models\backbones\facebookresearch\dinov2', model_name, trust_repo=True, source='local') # 加载DINOv2预训练模型
        # print(self.dino_model)
        self.device = torch.device(device)  # 数据计算所用设备‘cpu’ ‘cuda*’
        self.dino_model = self.dino_model.eval().to(self.device)  # 评估模式
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].register_forward_hook(self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].attn.qkv.register_forward_hook(self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out1 = output
        return _forward_hook


    def __call__(self, img: torch.Tensor) -> torch.Tensor:  # __cal__()可以作为直接调用该类的函数
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)

            if self.use_cls:
                res = self._hook_out1
            else:
                res = self._hook_out1[:, 1:, ...]

            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        return res


if __name__ == '__main__':
    x = torch.randn(96, 3, 224, 224).to('cuda')
    model = DinoV2(model_name='dinov2_vitg14',
               layer=31, facet="query", use_cls=False,
                 norm_descs=True, device="cuda", pretrained=True)
    print(model)
    print('-' * 70)

    # for i in range(len(model.dino_model.blocks)):
    #     if i <= 31:
    #         model.dino_model.blocks[i].requires_grad_(False)
    #     else:
    #         model.dino_model.blocks[i] = nn.Sequential()  # 设置为空的nn.Sequential()对象
    #
    #     model.dino_model.blocks[31].norm1.requires_grad_(True)  # 将requires_grad属性设置为True，即计算该部分的梯度
    #     model.dino_model.patch_embed.requires_grad_(False)  # requires_grad属性设置为False，即不计算该部分的梯度
    #     model.dino_model.norm = nn.Sequential()  # 设置为空的nn.Sequential()对象
    #     model.dino_model.head = nn.Sequential()  # 设置为空的nn.Sequential()对象
    #     model.norm = nn.Sequential()  # 设置为空的nn.Sequential()对象
    #     model.head = nn.Sequential()  # 设置为空的nn.Sequential()对象

    # print(model)
    r = model(x)
    # summary(model, input_size=(3, 224, 224), batch_size=96, device='cuda')
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')

