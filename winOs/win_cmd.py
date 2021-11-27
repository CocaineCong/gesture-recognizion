from winOs.pywin import *


def cmdPPT(handpose):
    # ppt = getPPT()
    # if not ppt:
    #     print("当前没有正在放映的ppt")
    #     return
    if handpose == "点击":
        click()
    elif handpose == "向左平移":
        turnLeft()
    elif handpose == "向右平移":
        turnRight()
    elif handpose == "缩放":
        esc()
    elif handpose == "放大":
        full()


def cmdPic(handpose):
    if handpose == "点击" or handpose == "放大":
        bigger()
    elif handpose == "抓取":
        fullscreen()
    elif handpose == "向左平移":
        turnLeft()
    elif handpose == "向右平移":
        turnRight()
    elif handpose == "缩放":
        smaller()
    elif handpose == "顺时针旋转":
        clockwise_acr()
    elif handpose == "逆时针旋转":
        counter_clockwise_acr()