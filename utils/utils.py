"""
    工具包
"""
import random
import cv2
import numpy as np
import torch
import math


torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)


def load_classes(path):
    """加载模型标签"""
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """在图像上画框"""
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def scale_coords(img_size, coords, img0_shape):
    """还原原图大小"""
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def parse_data_cfg(path):
    """读取配置文件"""
    print('data_cfg ： ', path)
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def get_k(p1, p2):
    """计算两点斜率"""
    x1, y1 = p1
    x2, y2 = p2
    return math.atan2(y2 - y1, x2 - x1)


def process_coordinate(coordinate: list, im0_shape):
    """获取手势框坐标坐标"""
    x_min, y_min, x_max, y_max = coordinate

    w_ = max(abs(x_max - x_min), abs(y_max - y_min)) * 1.1

    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    # 变方形
    x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

    x1 = np.clip(x1, 0, im0_shape[1] - 1)
    x2 = np.clip(x2, 0, im0_shape[1] - 1)
    y1 = np.clip(y1, 0, im0_shape[0] - 1)
    y2 = np.clip(y2, 0, im0_shape[0] - 1)
    return x1, x2, y1, y2


def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    """无伸缩缩放"""
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def process_data(img, img_size=416):
    """图像预处理"""
    # 填充使其变为方形
    img, _, _, _ = letterbox(img, height=img_size)
    # OpenCV读进来的图像,通道顺序为BGR， pytorch需要顺序为RGB，需转化
    # 样本数N, 通道数C, 高度H, 宽度W,pytorch输入顺序为N*C*H*W
    # transpose(2, 0, 1)将图片从H*W*C改为C*H*W的形式。
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    # 函数将一个内存不连续存储的数组转换为内存连续存储的数组,使得运行速度更快
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8->float32
    img /= 255.0    # 归一化
    return img


def draw_bd_handpose(img, hand, x, y):
    """连接手的21点"""
    colors = [(0, 215, 255), (255, 115, 55), (5, 255, 55), (25, 15, 255), (225, 15, 55)]
    cv2.line(img, (int(hand['0']['x']+x), int(hand['0']['y']+y)), (int(hand['1']['x']+x), int(hand['1']['y']+y)), colors[0], 2)
    cv2.line(img, (int(hand['1']['x']+x), int(hand['1']['y']+y)), (int(hand['2']['x']+x), int(hand['2']['y']+y)), colors[0], 2)
    cv2.line(img, (int(hand['2']['x']+x), int(hand['2']['y']+y)), (int(hand['3']['x']+x), int(hand['3']['y']+y)), colors[0], 2)
    cv2.line(img, (int(hand['3']['x']+x), int(hand['3']['y']+y)), (int(hand['4']['x']+x), int(hand['4']['y']+y)), colors[0], 2)
    cv2.line(img, (int(hand['0']['x']+x), int(hand['0']['y']+y)), (int(hand['5']['x']+x), int(hand['5']['y']+y)), colors[1], 2)
    cv2.line(img, (int(hand['5']['x']+x), int(hand['5']['y']+y)), (int(hand['6']['x']+x), int(hand['6']['y']+y)), colors[1], 2)
    cv2.line(img, (int(hand['6']['x']+x), int(hand['6']['y']+y)), (int(hand['7']['x']+x), int(hand['7']['y']+y)), colors[1], 2)
    cv2.line(img, (int(hand['7']['x']+x), int(hand['7']['y']+y)), (int(hand['8']['x']+x), int(hand['8']['y']+y)), colors[1], 2)
    cv2.line(img, (int(hand['0']['x']+x), int(hand['0']['y']+y)), (int(hand['9']['x']+x), int(hand['9']['y']+y)), colors[2], 2)
    cv2.line(img, (int(hand['9']['x']+x), int(hand['9']['y']+y)), (int(hand['10']['x']+x), int(hand['10']['y']+y)), colors[2], 2)
    cv2.line(img, (int(hand['10']['x']+x), int(hand['10']['y']+y)), (int(hand['11']['x']+x), int(hand['11']['y']+y)), colors[2], 2)
    cv2.line(img, (int(hand['11']['x']+x), int(hand['11']['y']+y)), (int(hand['12']['x']+x), int(hand['12']['y']+y)), colors[2], 2)
    cv2.line(img, (int(hand['0']['x']+x), int(hand['0']['y']+y)), (int(hand['13']['x']+x), int(hand['13']['y']+y)), colors[3], 2)
    cv2.line(img, (int(hand['13']['x']+x), int(hand['13']['y']+y)), (int(hand['14']['x']+x), int(hand['14']['y']+y)), colors[3], 2)
    cv2.line(img, (int(hand['14']['x']+x), int(hand['14']['y']+y)), (int(hand['15']['x']+x), int(hand['15']['y']+y)), colors[3], 2)
    cv2.line(img, (int(hand['15']['x']+x), int(hand['15']['y']+y)), (int(hand['16']['x']+x), int(hand['16']['y']+y)), colors[3], 2)
    cv2.line(img, (int(hand['0']['x']+x), int(hand['0']['y']+y)), (int(hand['17']['x']+x), int(hand['17']['y']+y)), colors[4], 2)
    cv2.line(img, (int(hand['17']['x']+x), int(hand['17']['y']+y)), (int(hand['18']['x']+x), int(hand['18']['y']+y)), colors[4], 2)
    cv2.line(img, (int(hand['18']['x']+x), int(hand['18']['y']+y)), (int(hand['19']['x']+x), int(hand['19']['y']+y)), colors[4], 2)
    cv2.line(img, (int(hand['19']['x']+x), int(hand['19']['y']+y)), (int(hand['20']['x']+x), int(hand['20']['y']+y)), colors[4], 2)