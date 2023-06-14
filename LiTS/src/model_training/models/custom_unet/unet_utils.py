import torch.nn as nn
from model_training.constants import ALPHA_VALUE

### Simple per-dim layer selection class
class LayerSet(object):
    def __init__(self, use_batchnorm:bool):
        self.conv  = nn.Conv2d
        self.tconv = nn.ConvTranspose2d
        self.norm  = nn.BatchNorm2d if use_batchnorm else nn.GroupNorm
        self.pool      = nn.MaxPool2d
        self.dropout   = nn.Dropout2d

class AuxiliaryPreparator(nn.Module):
    def __init__(self, filters_in, num_classes):
        super(AuxiliaryPreparator, self).__init__()
        self.get_aux_output = LayerSet(
            filters_in,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.out_act = nn.Softmax(dim=1) if num_classes > 1 else nn.Sigmoid()

    def forward(self, x):
        return self.out_act(self.get_aux_output(x))


class SqueezeExcitationRecalibration(nn.Module):
    def __init__(self, f_in, training_mode, value):
        super(SqueezeExcitationRecalibration, self).__init__()
        self.training_mode = training_mode
        self.pool = nn.AdaptiveAvgPool2d(1) if self.training_mode == "2D" else nn.AdaptiveAvgPool3d(1)
        self.recalib_weights = nn.Sequential(nn.Linear(f_in, int(f_in* value)),
                                             nn.LeakyReLU(ALPHA_VALUE), nn.Linear(int(f_in * value), f_in),
                                             nn.Sigmoid())
    def forward(self, x):
        bs, ch = x.size()[:2]
        recalib_weights = self.pool(x).view(bs,ch)
        if self.training_mode == '2D':
            recalib_weights = self.recalib_weights(recalib_weights).view(bs,ch,1,1)
        else:
            recalib_weights = self.recalib_weights(recalib_weights).view(bs,ch,1,1,1)
        return x * recalib_weights

class ResBlock(nn.Module):
    def __init__(self, filters_in, filters_out, dilate_val, use_batchnorm, reduce=4):
        super(ResBlock, self).__init__()
        layer_set = LayerSet(use_batchnorm)
        self.net = nn.Sequential(
            layer_set.conv(
                in_channels=filters_in,
                out_channels=filters_in//reduce,
                kernel_size=1,
                stride=1,
                padding=0),
            layer_set.norm(filters_in//reduce) if use_batchnorm else layer_set.norm(filters_in//reduce//4, filters_in//reduce),
            nn.LeakyReLU(ALPHA_VALUE),
            layer_set.conv(
                in_channels=filters_in//reduce,
                out_channels=filters_in//reduce,
                kernel_size=3,
                stride=1,
                padding=dilate_val,
                dilation=dilate_val),
            layer_set.norm(filters_in//reduce) if use_batchnorm else layer_set.norm(filters_in//reduce//4, filters_in//reduce),
            nn.LeakyReLU(ALPHA_VALUE),
            layer_set.conv(
                in_channels=filters_in//reduce,
                out_channels=filters_out,
                kernel_size=1,
                stride=1,
                padding=0)
            )

    def forward(self,x):
        return self.net(x)

class ResXBlock(nn.Module):
    def __init__(self, filters_in, filters_out, dilate_val, use_batchnorm):
        super(ResXBlock, self).__init__()
        group_reduce, cardinality = filters_in//8, np.clip(filters_in//8,None,32).astype(int)
        self.blocks = nn.ModuleList([ResBlock(
            filters_in=filters_in,
            filters_out=filters_out,
            dilate_val=dilate_val,
            use_batchnorm=use_batchnorm,
            reduce=group_reduce
            ) for _ in range(cardinality)])

    def forward(self,x):
        for i,block in enumerate(self.blocks):
            if i==0:
                out = block(x)
            else:
                out = out + block(x)
        return out