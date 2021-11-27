"""
    输出带操作
"""
from models.yolov3 import Yolov3
from models.resnet import resnet50
from utils.utils import process_data, scale_coords, process_coordinate, \
    draw_bd_handpose, get_k, plot_one_box, load_classes, parse_data_cfg
from utils.nms import non_max_suppression
import torch
import numpy as np
import cv2
import math
import os
from javaPredict import load_java_pkg, java_close
from winOs.win_cmd import *
from models.experimental import attempt_load


def get_hand_data(im0, mirror=True):
    if mirror:
        im0 = cv2.flip(im0, 1, dst=None)  # 水平镜像处理
    img = process_data(im0, 416)  # 数据预处理（用于传入yolo网络识别手的位置）

    # 图片检测
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    # pred, _ = yolov5_model(img)
    pred = yolo_model(img)[0]

    # 非极大值抑制nms
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
    if detections is None or len(detections) == 0:
        return im0, 0    # 未检测到手，直接跳过手势判断

    # 将结果映射到原图
    detections[:, :4] = scale_coords(416, detections[:, :4], im0.shape).round()
    results = []
    for *coordinate, conf, cls_conf, cls in detections:
        x1, x2, y1, y2 = process_coordinate(coordinate, im0.shape)
        hand_img = im0[y1:y2, x1:x2]

        img_width = hand_img.shape[1]
        img_height = hand_img.shape[0]
        # 输入图片预处理
        hand_img = cv2.resize(hand_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        hand_img = hand_img.astype(np.float32)
        hand_img = (hand_img - 128.0) / 256.0

        hand_img = hand_img.transpose(2, 0, 1)
        hand_img = torch.from_numpy(hand_img)
        hand_img = hand_img.unsqueeze_(0)

        if use_cuda:
            hand_img = hand_img.cuda()  # (bs, 3, h, w)
        pre_ = resnet50_model(hand_img.float())  # 模型推理
        output = pre_.cpu().detach().numpy()
        output = np.squeeze(output)

        pts_hand = {}  # 构建关键点连线可视化结构
        point = []
        for i in range(int(output.shape[0] / 2)):
            x = (output[i * 2 + 0] * float(img_width)) + x1
            y = (output[i * 2 + 1] * float(img_height)) + y1
            point.append((x, y))
            # cv2.putText(im0, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # 绘制关键点
            cv2.circle(im0, (int(x), int(y)), 3, (255, 50, 60), -1)
            cv2.circle(im0, (int(x), int(y)), 1, (255, 150, 180), -1)
            # pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x": x,
                "y": y,
            }
        draw_bd_handpose(im0, pts_hand, 0, 0)  # 绘制关键点连线
        flag = 0
        hand_label = "hand"
        if abs(180 - abs(get_k(point[0], point[3]) - get_k(point[3], point[4])) * 180 / math.pi) > 135:
            hand_label += "-1"
            flag += 1
        if abs(180 - abs(get_k(point[0], point[6]) - get_k(point[6], point[8])) * 180 / math.pi) > 135:
            hand_label += "-2"
            flag += 10
        if abs(180 - abs(get_k(point[0], point[10]) - get_k(point[10], point[12])) * 180 / math.pi) > 135:
            hand_label += "-3"
            flag += 100
        if abs(180 - abs(get_k(point[0], point[14]) - get_k(point[14], point[16])) * 180 / math.pi) > 135:
            hand_label += "-4"
            flag += 1000
        if abs(180 - abs(get_k(point[0], point[18]) - get_k(point[18], point[20])) * 180 / math.pi) > 135:
            hand_label += "-5"
            flag += 10000
        if flag == 0 or flag == 1:      # 握拳动作因人而异
            hand_label = "0"
        elif flag == 10 or flag == 11:
            hand_label = "1"
        else:
            hand_label = "2"
        label = '%s %.2f' % (hand_label, conf)
        plot_one_box(coordinate, im0, label=label, color=(100, 0, 100), line_thickness=2)
        results.append({
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "label": hand_label
        })
    return im0, results


if __name__ == '__main__':
    from config import conf_thres, nms_thres, cmd, data_cfg, \
        resnet50_model_path, yolov3_model_path, yolo_choose, \
        yolov5s_model_path, yolov5sm_model_path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 加载resnet50模型
    resnet50_model = resnet50(num_classes=42, img_size=256)
    resnet50_model = resnet50_model.to(device)
    resnet50_model.eval()
    if os.access(resnet50_model_path, os.F_OK):
        chkpt = torch.load(resnet50_model_path, map_location=device)
        resnet50_model.load_state_dict(chkpt)
    else:
        raise Exception("resnet50模型权重文件丢失，无法继续进行。")

    # 加载yolov5s模型
    if yolo_choose == "yolov3":
        # 加载yolov3模型
        classes = load_classes(parse_data_cfg(data_cfg)['names'])
        num_classes = len(classes)
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),
                   (373, 326)]  # yolov5先验框
        yolo_model = Yolov3(num_classes=num_classes, anchors=anchors)
        yolo_model = yolo_model.to(device)
        yolo_model.eval()
        if os.access(yolov3_model_path, os.F_OK):
            yolo_model.load_state_dict(torch.load(yolov3_model_path, map_location=device)['model'])
        else:
            raise Exception("yolov3模型权重文件丢失，无法继续进行。")
    elif yolo_choose == "yolov5sm":
        if os.access(yolov5sm_model_path, os.F_OK):
            yolo_model = attempt_load(yolov5sm_model_path, map_location=device)
        else:
            raise Exception("yolov5sm模型权重文件丢失，无法继续进行。")
    else:
        if os.access(yolov5s_model_path, os.F_OK):
            yolo_model = attempt_load(yolov5s_model_path, map_location=device)
        else:
            raise Exception("yolov5sm模型权重文件丢失，无法继续进行。")

    # 加载java环境,并返回预测对象
    recognizer, recorderInterface = load_java_pkg()

    # video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    video_capture = cv2.VideoCapture(0)                     # 选择默认摄像头
    # video_capture = cv2.VideoCapture('./video/3.mp4')     # 选择视频

    isSet = 0
    maxHeight = 0
    setFrame = 15
    middleLine = 240
    hands = {}
    d = {}
    with torch.no_grad():                           # 设置无梯度运行
        while True:
            ret, im0 = video_capture.read()         # 读取图片流
            if ret:
                im0, result = get_hand_data(im0, mirror=True)
                if result:
                    if isSet < setFrame and len(result) == 2:
                        """初始化（需要setFrame帧内手势最大位置变化不大）"""
                        newMaxHeight = max(result[0]["y"], result[1]["y"])
                        if abs(maxHeight - newMaxHeight) > 20:
                            maxHeight = max(result[0]["y"], result[1]["y"])
                            isSet = 0
                        else:
                            isSet += 1
                            if isSet == setFrame:
                                maxHeight -= 40     # 略大保证终止
                                recognizer.setDockLevel(maxHeight)
                                middleLine = (result[0]["x"] + result[0]["w"] // 2 + result[1]["x"] + result[1]["w"] // 2) // 2     # 取两手中点为中线
                                hands = {}
                                print("初始化成功,maxHeight:", maxHeight, "middleLine:", middleLine)
                    else:
                        """初始化完成后的判断"""
                        if not hands:
                            if len(result) > 1:
                                """有多个手，取边缘两只手进行比较，根据相对位置判断左右手"""
                                if result[0]["x"] + result[0]["w"] // 2 < result[-1]["x"] + result[-1]["w"] // 2:
                                    hands['0'] = result[0]
                                    hands['1'] = result[1]
                                else:
                                    hands['0'] = result[1]
                                    hands['1'] = result[0]
                        else:
                            t1 = None
                            t2 = None
                            m1 = 999999
                            m2 = 999999
                            if hands.get("0", None):    # 寻找与左手最近的手
                                v = hands.get("0")
                                for hand in result:
                                    """找出结果中和当前选择手最近的那个"""
                                    l = (hand['x'] - v['x']) * (hand['x'] - v['x']) + (hand['y'] - v['y']) * (hand['y'] - v['y'])
                                    if l < m1:
                                        m1 = l
                                        t1 = hand
                            if hands.get("1", None):    # 寻找与右手最近的手
                                v = hands.get("1")
                                for hand in result:
                                    """找出结果中和当前选择手最近的那个"""
                                    l = (hand['x'] - v['x']) * (hand['x'] - v['x']) + (hand['y'] - v['y']) * (hand['y'] - v['y'])
                                    if l < m2:
                                        m2 = l
                                        t2 = hand
                            if t1 == t2:
                                if m1 < m2:
                                    hands["0"] = t1
                                else:
                                    hands["1"] = t2
                            else:
                                if t1:
                                    hands["0"] = t1
                                if t2:
                                    hands["1"] = t2
                        if hands:
                            for (k, v) in hands.items():
                                ans = recorderInterface.addOne(
                                    v["x"],
                                    v["y"],
                                    v["w"],
                                    v["h"],
                                    int(v["label"]),
                                    int(k)
                                )
                                if ans:
                                    handpose = str(ans.getAction())
                                    if handpose == "CLICK":
                                        handpose = "点击"
                                    elif handpose == "PAN":
                                        if str(ans.getLocus()) == "LEFT":
                                            handpose = "向左平移"
                                        elif str(ans.getLocus()) == "RIGHT":
                                            handpose = "向右平移"
                                    elif handpose == "ZOOM":
                                        if str(ans.getLocus()) == "IN":
                                            handpose = "缩放"
                                        elif str(ans.getLocus()) == "OUT":
                                            handpose = "放大"
                                    elif handpose == "GRAB":
                                        handpose = "抓取"
                                    elif handpose == "PUNCH":
                                        handpose = "重置"
                                    elif handpose == "ROTATE":
                                        if str(ans.getLocus()) == "COUNTER_CLOCKWISE_ARC":
                                            handpose = "逆时针旋转"
                                        elif str(ans.getLocus()) == "CLOCKWISE_ARC":
                                            handpose = "顺时针旋转"
                                    if d.get("动作：" + handpose + ";手：" + str(ans.getHand())):
                                        d["动作：" + handpose + ";手：" + str(ans.getHand())] += 1
                                    else:
                                        d["动作：" + handpose + ";手：" + str(ans.getHand())] = 1
                                    print("动作：" + handpose + ";手：" + str(ans.getHand()))
                                    if handpose == "重置":
                                        isSet = 0
                                        maxHeight = 0
                                        hands = {}
                                        recognizer.setDockLevel(maxHeight)
                                    if cmd == 1:
                                        cmdPPT(handpose)
                                    elif cmd == 2:
                                        cmdPic(handpose)
                if isSet == setFrame:
                    """画初始线"""
                    cv2.line(im0, (0, maxHeight), (im0.shape[1], maxHeight), (255, 0, 0), 3)
                    cv2.line(im0, (middleLine, 0), (middleLine, im0.shape[0]), (255, 0, 0), 3)
                for (k, hand) in hands.items():
                    cv2.putText(im0, k, (hand["x"], hand['y']), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                cv2.namedWindow('image', 0)
                cv2.imshow("image", im0)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                elif key == 13:
                    isSet = 0
                    maxHeight = 0
                    hands = {}
                    recognizer.setDockLevel(maxHeight)
            else:
                break
        cv2.destroyAllWindows()
    java_close()
    print(d)

