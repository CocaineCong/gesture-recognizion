import win32gui
from winOs.win_mouse_key import *


def getPPT():
    hwnd_title = dict()
    def get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})
    win32gui.EnumWindows(get_all_hwnd, 0)
    ppt = None
    for h, t in hwnd_title.items():
        if " 幻灯片放映" in t:
            """找出当前正在放映的ppt"""
            ppt = {h: t}
    return ppt


def fullscreen():
    win32api.keybd_event(VK_CODE["f"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["f"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def turnLeft():
    win32api.keybd_event(VK_CODE["left_arrow"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["left_arrow"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def turnRight():
    win32api.keybd_event(VK_CODE["right_arrow"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["right_arrow"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def esc():
    win32api.keybd_event(VK_CODE["esc"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["esc"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def full():
    click()
    win32api.keybd_event(VK_CODE["F5"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["F5"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def bigger():
    win32api.keybd_event(VK_CODE["+"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["+"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def smaller():
    win32api.keybd_event(VK_CODE["-"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["-"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def clockwise_acr():
    win32api.keybd_event(VK_CODE["ctrl"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["alt"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["left_arrow"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["left_arrow"], 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(VK_CODE["ctrl"], 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(VK_CODE["alt"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def counter_clockwise_acr():
    win32api.keybd_event(VK_CODE["ctrl"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["alt"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["right_arrow"], 0, 0, 0)
    win32api.keybd_event(VK_CODE["right_arrow"], 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(VK_CODE["ctrl"], 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(VK_CODE["alt"], 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)


def click():
    mouse_click(500, 500)


if __name__ == '__main__':
    ppt = getPPT()
    print(ppt)


