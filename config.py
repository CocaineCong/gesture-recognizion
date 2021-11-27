# 检测置信度(置信度达到0.30视为目标物体，因为只有一个手的识别，置信度设置可以较低)
conf_thres = 0.30

# nms 阈值(非极大值抑制的阈值)
nms_thres = 0.5

# 两个模型和对应配置文件的地址
resnet50_model_path = './weights/resnet50.pth'
yolov3_model_path = './weights/yolov3.pt'
yolov5s_model_path = './weights/yolov5s.pt'
yolov5sm_model_path = './weights/yolov5sm.pt'
data_cfg = 'cfg/hand.data'

# yolo模型选择
yolo_choose = "yolov5sm"  # yolov3;yolov5sm

# java预测包地址
javaClassPath = './org/pkg/gesture-recognizion-1.6.3-jar-with-dependencies.jar'

# 0关闭控制；1控制ppt；2控制图片
cmd = 0