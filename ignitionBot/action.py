from pynput.mouse import Button, Controller
import win32api, win32con, win32gui
import autopy
import random

class action:
    def __init__(self):
        self.toplist = []
        self.winlist = []
        position = self.position()
        self.fold = (position[0]+372, position[1]+475, position[0]+400, position[1]+495)
        self.call = (position[0]+495, position[1]+475, position[0]+580, position[1]+492)    # and check
        self.raze = (position[0]+655, position[1]+475, position[0]+750, position[1]+492)    # and allin
        self.iamback = (position[0]+520, position[1]+470, position[0]+600, position[1]+495)

        
    def call_action(self):
        pos = self.position()
        x = random.randint(self.call[0], self.call[2])
        y = random.randint(self.call[1], self.call[3])
        autopy.mouse.smooth_move(x, y)
        autopy.mouse.click()
        autopy.mouse.move(pos[0], pos[1])

    def fold_action(self):
        pos = self.position()
        x = random.randint(self.fold[0], self.fold[2])
        y = random.randint(self.fold[1], self.fold[3])
        autopy.mouse.smooth_move(x, y)
        autopy.mouse.click()
        autopy.mouse.move(pos[0], pos[1])

    def i_am_back(self):
        pos = self.position()
        x = int(random.randint(self.iamback[0], self.iamback[2]))
        y = int(random.randint(self.iamback[1], self.iamback[3]))
        autopy.mouse.smooth_move(x, y)
        autopy.mouse.click()
        autopy.mouse.move(pos[0], pos[1])
        

    def raise_action(self):
        pos = self.position()
        x = random.randint(self.raze[0], self.raze[2])
        y = random.randint(self.raze[1], self.raze[3])
        autopy.mouse.smooth_move(x, y)
        autopy.mouse.click()
        autopy.mouse.move(pos[0], pos[1])
    
    def enum_cb(self, hwnd, results):
        self.winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    def position(self):
        win32gui.EnumWindows(self.enum_cb, self.toplist)
        firefox = [(hwnd, title) for hwnd, title in self.winlist if '.02' in title.lower()]
        # just grab the hwnd for first window matching firefox
        firefox = firefox[0]
        hwnd = firefox[0]

        win32gui.SetForegroundWindow(hwnd)
        return win32gui.GetWindowRect(hwnd)