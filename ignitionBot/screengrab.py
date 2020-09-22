from PIL import ImageGrab
import win32gui


class Screengrab():
    def __init__(self):
        self.toplist = []
        self.winlist = []

    def enum_cb(self, hwnd, results):
        self.winlist.append((hwnd, win32gui.GetWindowText(hwnd)))


    def screenshot(self):
        try:
            bbox = self.position()
        except:
            return ''
        img = ImageGrab.grab(bbox)
        return img
    
    def position(self):
        win32gui.EnumWindows(self.enum_cb, self.toplist)
        firefox = [(hwnd, title) for hwnd, title in self.winlist if '.02' in title.lower()]
        # just grab the hwnd for first window matching firefox
        firefox = firefox[0]
        hwnd = firefox[0]

        win32gui.SetForegroundWindow(hwnd)
        return win32gui.GetWindowRect(hwnd)
