from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear
from paddle.regularizer import L2Decay


from abc import ABC
from paddle import nn
import re


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class TheseusLayer(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(TheseusLayer, self).__init__()
        self.res_dict = {}
        self.res_name = self.full_name()

    # stop doesn't work when stop layer has a parallel branch.
    def stop_after(self, stop_layer_name: str):
        after_stop = False
        for layer_i in self._sub_layers:
            if after_stop:
                self._sub_layers[layer_i] = Identity()
                continue
            layer_name = self._sub_layers[layer_i].full_name()
            if layer_name == stop_layer_name:
                after_stop = True
                continue
            if isinstance(self._sub_layers[layer_i], TheseusLayer):
                after_stop = self._sub_layers[layer_i].stop_after(
                    stop_layer_name)
        return after_stop

    def update_res(self, return_patterns):
        for return_pattern in return_patterns:
            pattern_list = return_pattern.split(".")
            if not pattern_list:
                continue
            sub_layer_parent = self
            while len(pattern_list) > 1:
                if '[' in pattern_list[0]:
                    sub_layer_name = pattern_list[0].split('[')[0]
                    sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
                    sub_layer_parent = getattr(sub_layer_parent, sub_layer_name)[sub_layer_index]
                else:
                    sub_layer_parent = getattr(sub_layer_parent, pattern_list[0],
                                               None)
                    if sub_layer_parent is None:
                        break
                if isinstance(sub_layer_parent, WrapLayer):
                    sub_layer_parent = sub_layer_parent.sub_layer
                pattern_list = pattern_list[1:]
            if sub_layer_parent is None:
                continue
            if '[' in pattern_list[0]:
                sub_layer_name = pattern_list[0].split('[')[0]
                sub_layer_index = pattern_list[0].split('[')[1].split(']')[0]
                sub_layer = getattr(sub_layer_parent, sub_layer_name)[sub_layer_index]
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                getattr(sub_layer_parent, sub_layer_name)[sub_layer_index] = sub_layer
            else:
                sub_layer = getattr(sub_layer_parent, pattern_list[0])
                if not isinstance(sub_layer, TheseusLayer):
                    sub_layer = wrap_theseus(sub_layer)
                setattr(sub_layer_parent, pattern_list[0], sub_layer)

            sub_layer.res_dict = self.res_dict
            sub_layer.res_name = return_pattern
            sub_layer.register_forward_post_hook(sub_layer._save_sub_res_hook)

    def _save_sub_res_hook(self, layer, input, output):
        self.res_dict[self.res_name] = output

    def _return_dict_hook(self, layer, input, output):
        res_dict = {"output": output}
        for res_key in list(self.res_dict):
            res_dict[res_key] = self.res_dict.pop(res_key)
        return res_dict

    def replace_sub(self, layer_name_pattern, replace_function,
                    recursive=True):
        for layer_i in self._sub_layers:
            layer_name = self._sub_layers[layer_i].full_name()
            if re.match(layer_name_pattern, layer_name):
                self._sub_layers[layer_i] = replace_function(self._sub_layers[
                    layer_i])
            if recursive:
                if isinstance(self._sub_layers[layer_i], TheseusLayer):
                    self._sub_layers[layer_i].replace_sub(
                        layer_name_pattern, replace_function, recursive)
                elif isinstance(self._sub_layers[layer_i],
                                (nn.Sequential, nn.LayerList)):
                    for layer_j in self._sub_layers[layer_i]._sub_layers:
                        self._sub_layers[layer_i]._sub_layers[
                            layer_j].replace_sub(layer_name_pattern,
                                                 replace_function, recursive)

    '''
    example of replace function:
    def replace_conv(origin_conv: nn.Conv2D):
        new_conv = nn.Conv2D(
            in_channels=origin_conv._in_channels,
            out_channels=origin_conv._out_channels,
            kernel_size=origin_conv._kernel_size,
            stride=2
        )
        return new_conv
        '''


class WrapLayer(TheseusLayer):
    def __init__(self, sub_layer):
        super(WrapLayer, self).__init__()
        self.sub_layer = sub_layer

    def forward(self, *inputs, **kwargs):
        return self.sub_layer(*inputs, **kwargs)


def wrap_theseus(sub_layer):
    wrapped_layer = WrapLayer(sub_layer)
    return wrapped_layer


NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ]
}
# first conv output channel number in MobileNetV3
STEM_CONV_NUMBER = 16
# last second conv output channel for "small"
LAST_SECOND_CONV_SMALL = 576
# last second conv output channel for "large"
LAST_SECOND_CONV_LARGE = 960
# last conv output channel number for "large" and "small"
LAST_CONV = 1280


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class MobileNetV3(TheseusLayer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        scale: float=1.0. The coefficient that controls the size of network parameters.
        class_num: int=1000. The number of classes.
        inplanes: int=16. The output channel number of first convolution layer.
        class_squeeze: int=960. The output channel number of penultimate convolution layer.
        class_expand: int=1280. The output channel number of last convolution layer.
        dropout_prob: float=0.2.  Probability of setting units to zero.
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 config=NET_CONFIG["large"],
                 scale=1.0,
                 class_num=1000,
                 inplanes=STEM_CONV_NUMBER,
                 class_squeeze=LAST_SECOND_CONV_LARGE,
                 class_expand=LAST_CONV,
                 dropout_prob=0.2):
        super().__init__()

        self.cfg = config
        self.scale = scale
        self.inplanes = inplanes
        self.class_squeeze = class_squeeze
        self.class_expand = class_expand
        self.class_num = class_num

        self.conv = ConvBNLayer(
            in_c=3,
            out_c=_make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish")

        self.blocks = nn.Sequential(*[
            ResidualUnit(
                in_c=_make_divisible(self.inplanes * self.scale if i == 0 else
                                     self.cfg[i - 1][2] * self.scale),
                mid_c=_make_divisible(self.scale * exp),
                out_c=_make_divisible(self.scale * c),
                filter_size=k,
                stride=s,
                use_se=se,
                act=act) for i, (k, exp, c, se, act, s) in enumerate(self.cfg)
        ])

    def forward(self, x):
        x = self.conv(x)
        x1 = x
        y1 = y2 = y3 = y4 = None
        k = 0
        for i in self.blocks:
            x = i(x)
            k += 1
            if k == 3:
                y1 = x
            if k==6:
                y2 = x
            if k==12:
                y3 = x
            y4 = x
        return x1, y1, y2, y3, y4

    def init_weight(self):
        pass


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None):
        super().__init__()

        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


# nn.Hardsigmoid can't transfer "slope" and "offset" in nn.functional.hardsigmoid
class Hardsigmoid(TheseusLayer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(
            x, slope=self.slope, offset=self.offset)


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


def MobileNetV3_0_5():
    model = MobileNetV3(scale=0.5)
    return model
