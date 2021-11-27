import jpype
import os
from config import javaClassPath


def load_java_pkg():
    # 加载java环境
    # javaClassPath = './org/pkg/gesture-recognizion-1.6.3-jar-with-dependencies.jar'
    jpype.startJVM(classpath=['jars/*'])
    jarpath = os.path.join(os.path.abspath('.'), 'org/pkg')
    jpype.addClassPath(javaClassPath)
    # 调用预测类
    RecorderInterface = jpype.JClass("top.chenzhimeng.gesturerecognizion.RecorderInterface")
    Recognizer = jpype.JClass("top.chenzhimeng.gesturerecognizion.component.RecorderHandler")
    Monitor = jpype.JClass("top.chenzhimeng.gesturerecognizion.component.Monitor")
    Float = jpype.JClass("java.lang.Float")
    Integer = jpype.JClass("java.lang.Integer")
    # 参数设置
    monitor = Monitor.builder()\
        .dockLevel(Integer(0))\
        .dockJudgmentPointsNum(Integer(3))\
        .build()
    recognizer = Recognizer.builder()\
        .headRatio(Float(0.2))\
        .tailRatio(Float(0.35))\
        .lineDegree(Integer(30))\
        .rotateDegreeRange([31, 180])\
        .clickFingerRatio(Float(0.5))\
        .panPalmRatio(Float(0.5))\
        .panLenThreshold(Integer(68))\
        .monitor(monitor)\
        .build()

    RecorderInterface.debug = False     # 生产环境下关闭debug
    recorderInterface = RecorderInterface(recognizer)
    return recognizer, recorderInterface


def java_close():
    jpype.shutdownJVM()