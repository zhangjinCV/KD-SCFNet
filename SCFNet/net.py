import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal, Constant
from backbones.ResNet import ResNet50
from backbones.SwinT import SwinTransformer_B
from backbones.MobileNetV3 import MobileNetV3_0_5
kaiming_normal_init = KaimingNormal()
constant_value_1 = Constant(value=1.0)
constant_value_0 = Constant(value=0.0)
# -------MY MODEL----------#


def weight_init(module):
    for n, m in module.named_children():
       # print('initialize: ' + n)
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)

        elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)):
            constant_value_1(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)
        elif isinstance(m, nn.Linear):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2D):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2D):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.MaxPool2D):
            pass
        else:
            m.init_weight()


class MappingModule(nn.Layer):
    def __init__(self, type, out_c):
        super(MappingModule, self).__init__()
        if type == 'M3_0.5':
            nums = [16, 24, 56, 80]
        elif type == 'R34':
            nums = [64, 128, 256, 512]
        elif type == 'ST':
            nums = [128, 256, 512, 1024]
        else:
            nums = [256, 512, 1024, 2048]
        self.cv1 = nn.Sequential(
            nn.Conv2D(nums[0], out_c, 3, 1, 1),
            nn.BatchNorm2D(out_c),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(nums[1], out_c, 3, 1, 1),
            nn.BatchNorm2D(out_c),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(nums[2], out_c, 3, 1, 1),
            nn.BatchNorm2D(out_c),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(nums[3], out_c, 3, 1, 1),
            nn.BatchNorm2D(out_c),
            nn.ReLU()
        )

    def forward(self, out2, out3, out4, out5):
        out2 = self.cv1(out2)
        out3 = self.cv2(out3)
        out4 = self.cv3(out4)
        out5 = self.cv4(out5)
        return out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


class SCFM(nn.Layer):
    def __init__(self, in_c):
        super(SCFM, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2D(in_c),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2D(in_c),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2D(in_c),
            nn.ReLU()
        )

    def forward(self, fl, fh, fd):
        fdh = F.interpolate(fd, size=fh.shape[2:], mode='bilinear')
        fdh = fh * fdh
        fdh = self.cv1(fh + fdh) 

        fhl = F.interpolate(fh, size=fl.shape[2:], mode='bilinear')
        fhl = fl * fhl
        fhl = self.cv2(fl + fhl)

        fdh = F.interpolate(fdh, size=fl.shape[2:], mode='bilinear')
        fll = self.cv3(fhl + fdh)
        return fll

    def init_weight(self):
        weight_init(self)


class SCFNet(nn.Layer):
    def __init__(self, cfg):
        super(SCFNet, self).__init__()
        self.cfg           =  cfg
        self.in_c          =  64
        backbone_type = cfg.model_type
        if backbone_type   == 'R34':
            self.bkbone = ResNet34()
            self.bkbone.load_dict(paddle.vision.models.resnet34(True).state_dict())
        elif backbone_type == 'R50':
            self.bkbone    = ResNet50()
            self.bkbone.load_dict(paddle.vision.models.resnet50(True).state_dict())
        elif backbone_type == 'M3_0.5':
            self.bkbone = MobileNetV3_0_5()
            self.bkbone.load_dict(paddle.load('./backbones/MobileNetV3_large_x0_5_pretrained.pdparams'))
            self.in_c = 16
        elif backbone_type == 'ST':
            self.bkbone = SwinTransformer_B()
            self.bkbone.load_dict(paddle.load('./backbones/SwinT_B.pdparams'))
        else:
            raise ValueError("the backbone_type should be one of 'R50', 'M3_0.5', 'ST")
        self.mp = MappingModule(backbone_type, self.in_c)
        self.cv1 = nn.Sequential(nn.Conv2D(self.in_c, self.in_c, 3, 1, 5, dilation=5), nn.BatchNorm2D(self.in_c), nn.ReLU())
        self.de1 = SCFM(self.in_c)
        self.de2 = SCFM(self.in_c)
        self.de3 = SCFM(self.in_c)
        self.linear2 = nn.Conv2D(self.in_c, 1, 3, 1, 1)
        self.linear3 = nn.Conv2D(self.in_c, 1, 3, 1, 1)
        self.linear4 = nn.Conv2D(self.in_c, 1, 3, 1, 1)
        self.linear5 = nn.Conv2D(self.in_c, 1, 3, 1, 1)
        for p in self.bkbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0
        self.init_weight()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.bkbone(x)
        x2, x3, x4, x5     = self.mp(x2, x3, x4, x5)
        x5                 = self.cv1(x5)
        x4                 = self.de3(x4, x5, x5)
        x3                 = self.de2(x3, x4, x5)
        x2                 = self.de1(x2, x3, x5)
        x2                 = F.interpolate(self.linear2(x2), mode='bilinear', size=x.shape[2:])
        x3                 = F.interpolate(self.linear3(x3), mode='bilinear', size=x.shape[2:])
        x4                 = F.interpolate(self.linear4(x4), mode='bilinear', size=x.shape[2:])
        x5                 = F.interpolate(self.linear5(x5), mode='bilinear', size=x.shape[2:])
        return x2, x3, x4, x5

    def init_weight(self):
        weight_init(self)


if __name__ == '__main__':
    from train_FS import cag
    f4 = SCFNet(cag)
    total_params = sum(p.numel() for p in f4.parameters())
    print('total params : ', total_params)
    FLOPs = paddle.flops(f4, [1, 3, 352, 352],
                         print_detail=False)
    print(FLOPs)
