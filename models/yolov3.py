import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn


# reference:
# https://github.com/ultralytics/yolov3/blob/master/models.py
# https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/blob/master/yolo/vedanet/network/backbone/brick/darknet53.py
# network structure https://blog.csdn.net/u010397980/article/details/85058630

flag_yolo_structure = False # True 查看 相关的网络 log

class Conv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
            if flag_yolo_structure:
                print('------------------->>>> Conv2dBatchLeaky isinstance')
        else:
            self.padding = int(kernel_size/2)

        self.leaky_slope = leaky_slope
        # Layer
        # LeakyReLU : y = max(0, x) + leaky_slope*min(0,x)
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ResBlockSum(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
            Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
            )

    def forward(self, x):
        return x + self.block(x)

class HeadBody(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadBody, self).__init__()

        self.layer = nn.Sequential(
            Conv2dBatchLeaky(in_channels, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels*2, 3, 1),
            Conv2dBatchLeaky(out_channels*2, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels*2, 3, 1),
            Conv2dBatchLeaky(out_channels*2, out_channels, 1, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

# default anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes
        self.img_size = 0
        if flag_yolo_structure:
            print('init YOLOLayer ------ >>> ')
            print('anchors  : ',self.anchors)
            print('nA       : ',self.nA)
            print('nC       : ',self.nC)
            print('img_size : ',self.img_size)

    def forward(self, p, img_size, var=None):# p : feature map
        bs, nG = p.shape[0], p.shape[-1] # batch_size , grid
        if flag_yolo_structure:
            print('bs, nG --->>> ',bs, nG)
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, xywh + confidence + classes)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  #  prediction

        if self.training:
            return p
        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            io[..., :4] *= self.stride
            if self.nC == 1:
                io[..., 5] = 1  # single-class model
            # flatten prediction, reshape from [bs, nA, nG, nG, nC] to [bs, nA * nG * nG, nC]
            return io.view(bs, -1, 5 + self.nC), p

def create_grids(self, img_size, nG, device='cpu'):
    # self.nA : len(anchors)  # number of anchors (3)
    # self.nC : nC  # number of classes
    # nG : feature map grid  13*13  26*26 52*52
    self.img_size = img_size
    self.stride = img_size / nG
    if flag_yolo_structure:
        print('create_grids stride : ',self.stride)

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)
    if flag_yolo_structure:
        print('grid_x : ',grid_x.size(),grid_x)
        print('grid_y : ',grid_y.size(),grid_y)
        print('grid_xy : ',self.grid_xy.size(),self.grid_xy)

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride # 基于 stride 的归一化
    # print('self.anchor_vecself.anchor_vecself.anchor_vec:',self.anchor_vec)
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)
    self.nG = torch.FloatTensor([nG]).to(device)


def get_yolo_layer_index(module_list):
    yolo_layer_index = []
    for index, l in enumerate(module_list):
        try:
            a = l[0].img_size and l[0].nG  # only yolo layer need img_size and nG
            yolo_layer_index.append(index)
        except:
            pass
    assert len(yolo_layer_index) > 0, "can not find yolo layer"
    return yolo_layer_index


# ----------------------yolov3------------------------

class Yolov3(nn.Module):
    def __init__(self, num_classes=80, anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]):
        super().__init__()
        anchor_mask1 = [i for i in range(2 * len(anchors) // 3, len(anchors), 1)]  # [6, 7, 8]
        anchor_mask2 = [i for i in range(len(anchors) // 3, 2 * len(anchors) // 3, 1)]  # [3, 4, 5]
        anchor_mask3 = [i for i in range(0, len(anchors) // 3, 1)]  # [0, 1, 2]
        if flag_yolo_structure:
            print('anchor_mask1 ： ',anchor_mask1) # 大物体 anchor
            print('anchor_mask2 ： ',anchor_mask2) # 中物体 anchor
            print('anchor_mask3 ： ',anchor_mask3) # 小物体 anchor

        # Network
        # OrderedDict 是 dict 的子类，其最大特征是，它可以“维护”添加 key-value 对的顺序
        layer_list = []

        '''
        ******      Conv2dBatchLeaky       *****
        op : Conv2d,BatchNorm2d,LeakyReLU
        inputs : in_channels, out_channels, kernel_size, stride, leaky_slope
        '''

        '''
        ******      ResBlockSum ******
        op : Conv2dBatchLeaky * 2 + x
        inputs : nchannels
        '''
        # list 0
        layer_list.append(OrderedDict([
            ('0_stage1_conv', Conv2dBatchLeaky(3, 32, 3, 1, 1)),  # 416 x 416 x 32        # Convolutional

            ("0_stage2_conv", Conv2dBatchLeaky(32, 64, 3, 2)),  # 208 x 208 x 64          # Convolutional
            ("0_stage2_ressum1", ResBlockSum(64)),                                        # Convolutional*2 + Resiudal

            ("0_stage3_conv", Conv2dBatchLeaky(64, 128, 3, 2)),  # 104 x 104 128          # Convolutional
            ("0_stage3_ressum1", ResBlockSum(128)),
            ("0_stage3_ressum2", ResBlockSum(128)),                                       # (Convolutional*2 + Resiudal)**2

            ("0_stage4_conv", Conv2dBatchLeaky(128, 256, 3, 2)),  # 52 x 52 x 256         # Convolutional
            ("0_stage4_ressum1", ResBlockSum(256)),
            ("0_stage4_ressum2", ResBlockSum(256)),
            ("0_stage4_ressum3", ResBlockSum(256)),
            ("0_stage4_ressum4", ResBlockSum(256)),
            ("0_stage4_ressum5", ResBlockSum(256)),
            ("0_stage4_ressum6", ResBlockSum(256)),
            ("0_stage4_ressum7", ResBlockSum(256)),
            ("0_stage4_ressum8", ResBlockSum(256)),  # 52 x 52 x 256 output_feature_0      (Convolutional*2 + Resiudal)**8
            ]))
        # list 1
        layer_list.append(OrderedDict([
            ("1_stage5_conv", Conv2dBatchLeaky(256, 512, 3, 2)),  # 26 x 26 x 512         # Convolutional
            ("1_stage5_ressum1", ResBlockSum(512)),
            ("1_stage5_ressum2", ResBlockSum(512)),
            ("1_stage5_ressum3", ResBlockSum(512)),
            ("1_stage5_ressum4", ResBlockSum(512)),
            ("1_stage5_ressum5", ResBlockSum(512)),
            ("1_stage5_ressum6", ResBlockSum(512)),
            ("1_stage5_ressum7", ResBlockSum(512)),
            ("1_stage5_ressum8", ResBlockSum(512)),  # 26 x 26 x 512 output_feature_1     # (Convolutional*2 + Resiudal)**8
            ]))

        '''
        ******      HeadBody      ******
        op : Conv2dBatchLeaky * 5
        inputs : in_channels, out_channels
        '''
        # list 2
        layer_list.append(OrderedDict([
            ("2_stage6_conv", Conv2dBatchLeaky(512, 1024, 3, 2)),  # 13 x 13 x 1024      # Convolutional
            ("2_stage6_ressum1", ResBlockSum(1024)),
            ("2_stage6_ressum2", ResBlockSum(1024)),
            ("2_stage6_ressum3", ResBlockSum(1024)),
            ("2_stage6_ressum4", ResBlockSum(1024)),  # 13 x 13 x 1024 output_feature_2 # (Convolutional*2 + Resiudal)**4
            ("2_headbody1", HeadBody(in_channels=1024, out_channels=512)), # 13 x 13 x 512  # Convalutional Set = Conv2dBatchLeaky * 5
            ]))
        # list 3
        layer_list.append(OrderedDict([
            ("3_conv_1", Conv2dBatchLeaky(in_channels=512, out_channels=1024, kernel_size=3, stride=1)),
            ("3_conv_2", nn.Conv2d(in_channels=1024, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True)),
        ])) # predict one
        # list 4
        layer_list.append(OrderedDict([
            ("4_yolo", YOLOLayer([anchors[i] for i in anchor_mask1], num_classes))
        ])) # 3*((x, y, w, h, confidence) + classes )

        # list 5
        layer_list.append(OrderedDict([
            ("5_conv", Conv2dBatchLeaky(512, 256, 1, 1)),
            ("5_upsample", Upsample(scale_factor=2)),
        ]))
        # list 6
        layer_list.append(OrderedDict([
            ("6_head_body2", HeadBody(in_channels=768, out_channels=256)) # Convalutional Set = Conv2dBatchLeaky * 5
        ]))
        # list 7
        layer_list.append(OrderedDict([
            ("7_conv_1", Conv2dBatchLeaky(in_channels=256, out_channels=512, kernel_size=3, stride=1)),
            ("7_conv_2", nn.Conv2d(in_channels=512, out_channels=len(anchor_mask2) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True)),
        ])) # predict two
        # list 8
        layer_list.append(OrderedDict([
            ("8_yolo", YOLOLayer([anchors[i] for i in anchor_mask2], num_classes))
        ])) # 3*((x, y, w, h, confidence) + classes )
        # list 9
        layer_list.append(OrderedDict([
            ("9_conv", Conv2dBatchLeaky(256, 128, 1, 1)),
            ("9_upsample", Upsample(scale_factor=2)),
        ]))
        # list 10
        layer_list.append(OrderedDict([
            ("10_head_body3", HeadBody(in_channels=384, out_channels=128))  # Convalutional Set = Conv2dBatchLeaky * 5
        ]))
        # list 11
        layer_list.append(OrderedDict([
            ("11_conv_1", Conv2dBatchLeaky(in_channels=128, out_channels=256, kernel_size=3, stride=1)),
            ("11_conv_2", nn.Conv2d(in_channels=256, out_channels=len(anchor_mask3) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True)),
        ])) # predict three
        # list 12
        layer_list.append(OrderedDict([
            ("12_yolo", YOLOLayer([anchors[i] for i in anchor_mask3], num_classes))
        ])) # 3*((x, y, w, h, confidence) + classes )
        # nn.ModuleList类似于pytho中的list类型，只是将一系列层装入列表，并没有实现forward()方法，因此也不会有网络模型产生的副作用
        self.module_list = nn.ModuleList([nn.Sequential(i) for i in layer_list])
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)
        if flag_yolo_structure:
            print('yolo_layer : ',len(layer_list),'\n')
            print(self.module_list[4])
            print(self.module_list[8])
            print(self.module_list[12])

        # print('self.module_list  -------->>> ',self.module_list)
        # print('self.yolo_layer_index  -------->>> ',self.yolo_layer_index)

    def forward(self, x):
        img_size = x.shape[-1]
        if flag_yolo_structure:
            print('forward img_size : ',img_size,x.shape)
        output = []

        x = self.module_list[0](x)
        x_route1 = x
        x = self.module_list[1](x)
        x_route2 = x
        x = self.module_list[2](x)

        yolo_head = self.module_list[3](x)
        if flag_yolo_structure:
            print('mask1 yolo_head : ',yolo_head.size())
        yolo_head_out_13x13 = self.module_list[4][0](yolo_head, img_size)
        output.append(yolo_head_out_13x13)

        x = self.module_list[5](x)
        x = torch.cat([x, x_route2], 1)
        x = self.module_list[6](x)

        yolo_head = self.module_list[7](x)
        if flag_yolo_structure:
            print('mask2 yolo_head : ',yolo_head.size())
        yolo_head_out_26x26 = self.module_list[8][0](yolo_head, img_size)
        output.append(yolo_head_out_26x26)

        x = self.module_list[9](x)
        x = torch.cat([x, x_route1], 1)
        x = self.module_list[10](x)

        yolo_head = self.module_list[11](x)
        if flag_yolo_structure:
            print('mask3 yolo_head : ',yolo_head.size())
        yolo_head_out_52x52 = self.module_list[12][0](yolo_head, img_size)
        output.append(yolo_head_out_52x52)

        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


# ----------------------yolov3 tiny------------------------

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Yolov3Tiny(nn.Module):
    def __init__(self, num_classes=80, anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]):
        super(Yolov3Tiny, self).__init__()

        anchor_mask1 = [i for i in range(len(anchors) // 2, len(anchors), 1)]  # [3, 4, 5]
        anchor_mask2 = [i for i in range(0, len(anchors) // 2, 1)]  # [0, 1, 2]

        layer_list = []
        layer_list.append(OrderedDict([
            # layer 0
            ("conv_0", nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_0", nn.BatchNorm2d(16)),
            ("leaky_0", nn.LeakyReLU(0.1)),
            # layer 1
            ("maxpool_1", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 2
            ("conv_2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_2", nn.BatchNorm2d(32)),
            ("leaky_2", nn.LeakyReLU(0.1)),
            # layer 3
            ("maxpool_3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 4
            ("conv_4", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_4", nn.BatchNorm2d(64)),
            ("leaky_4", nn.LeakyReLU(0.1)),
            # layer 5
            ("maxpool_5", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 6
            ("conv_6", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_6", nn.BatchNorm2d(128)),
            ("leaky_6", nn.LeakyReLU(0.1)),
            # layer 7
            ("maxpool_7", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 8
            ("conv_8", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_8", nn.BatchNorm2d(256)),
            ("leaky_8", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 9
            ("maxpool_9", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 10
            ("conv_10", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_10", nn.BatchNorm2d(512)),
            ("leaky_10", nn.LeakyReLU(0.1)),
            # layer 11
            ('_debug_padding_11', nn.ZeroPad2d((0, 1, 0, 1))),
            ("maxpool_11", nn.MaxPool2d(kernel_size=2, stride=1, padding=0)),
            # layer 12
            ("conv_12", nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_12", nn.BatchNorm2d(1024)),
            ("leaky_12", nn.LeakyReLU(0.1)),
            # layer 13
            ("conv_13", nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_13", nn.BatchNorm2d(256)),
            ("leaky_13", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 14
            ("conv_14", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_14", nn.BatchNorm2d(512)),
            ("leaky_14", nn.LeakyReLU(0.1)),
            # layer 15
            ("conv_15",
             nn.Conv2d(in_channels=512, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True)),
        ]))

        # layer 16
        anchor_tmp1 = [anchors[i] for i in anchor_mask1]
        layer_list.append(OrderedDict([("yolo_16", YOLOLayer(anchor_tmp1, num_classes))]))

        # layer 17
        layer_list.append(OrderedDict([("route_17", EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 18
            ("conv_18", nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_18", nn.BatchNorm2d(128)),
            ("leaky_18", nn.LeakyReLU(0.1)),
            # layer 19
            ("upsample_19", Upsample(scale_factor=2)),
        ]))

        # layer 20
        layer_list.append(OrderedDict([('route_20', EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 21
            ("conv_21", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_21", nn.BatchNorm2d(256)),
            ("leaky_21", nn.LeakyReLU(0.1)),
            # layer 22
            ("conv_22",
             nn.Conv2d(in_channels=256, out_channels=len(anchor_mask2) * (num_classes + 5), kernel_size=1, stride=1, padding=0, bias=True)),
        ]))

        # layer 23
        anchor_tmp2 = [anchors[i] for i in anchor_mask2]
        layer_list.append(OrderedDict([("yolo_23", YOLOLayer(anchor_tmp2, num_classes))]))

        self.module_list = nn.ModuleList([nn.Sequential(layer) for layer in layer_list])
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)

    def forward(self, x):
        img_size = x.shape[-1]
        output = []

        x = self.module_list[0](x)  # layer0 to layer8
        x_route8 = x
        x = self.module_list[1](x)  # layer9 to layer13
        x_route13 = x
        x = self.module_list[2](x)  # layer14, layer15
        x = self.module_list[3][0](x, img_size)  # yolo_16
        output.append(x)
        x = self.module_list[5](x_route13)  # layer18, layer19
        x = torch.cat([x, x_route8], 1)  # route
        x = self.module_list[7](x)  # layer21, layer22
        x = self.module_list[8][0](x, img_size)  # yolo_23
        output.append(x)

        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


if __name__ == "__main__":
    dummy_input = torch.Tensor(5, 3, 416, 416)
    model = Yolov3(num_classes=80)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        # print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: " + str(k))
    print("-----------yolo layer")
    for index in model.yolo_layer_index:
        print(model.module_list[index])

    print("-----------train")
    model.train()
    for res in model(dummy_input):
        print("res:", np.shape(res))

    print("-----------eval")
    model.eval()
    inference_out, train_out = model(dummy_input)
    print("inference_out:", np.shape(inference_out))
    for o in train_out:
        print("train_out:", np.shape(o))
