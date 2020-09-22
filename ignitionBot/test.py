import win32gui
import numpy as np
import pytesseract
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from PIL import Image, ImageFilter, ImageOps

    # card1, card2, button, equity, bet
p1 = [(450, 355, 480, 405), (490, 355, 520, 405), (525, 370, 540, 385), (465, 405, 535, 430), (470, 337, 530, 354)]
p2 = [(175, 305, 205, 355), (210, 305, 240, 355), (275, 360, 290, 375), (190, 355, 260, 380), (267, 317, 327, 334)]
p3 = [(175, 145, 205, 195), (210, 145, 240, 195), (275, 195, 290, 210), (190, 190, 260, 215), (288, 219, 348, 236)]
p4 = [(450, 90, 480, 135), (490, 90, 515, 135), (550, 130, 565, 145), (465, 137, 535, 162), (560, 168, 610, 185)]
p5 = [(725, 145, 755, 195), (760, 145, 790, 195), (685, 195, 700, 210), (740, 190, 810, 215), (665, 220, 715, 237)]
p6 = [(725, 310, 755, 360), (760, 310, 790, 360), (685, 360, 700, 375), (740, 355, 810, 380), (690, 317, 740, 334)]

plyrs = [p1, p2, p3, p4, p5, p6]
img = Image.open("2 - Pre-flop.png")
def test(bx, img):
    def binarize_array(image, threshold=200):
        """Binarize a numpy array."""
        numpy_array = np.array(image)
        for i in range(len(numpy_array)):
            for j in range(len(numpy_array[0])):
                if numpy_array[i][j] > threshold:
                    numpy_array[i][j] = 255
                else:
                    numpy_array[i][j] = 0
        return Image.fromarray(numpy_array)

    img_orig = ImageOps.invert(img)
    lst = []
    basewidth = 125
    wpercent = (basewidth / float(img_orig.size[0]))
    hsize = int((float(img_orig.size[1]) * float(wpercent)))
    img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
    img_resized = binarize_array(img_resized, 90)
    img_resized.show()
    lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=CALL$0123456789.'))
    if("CALL" in lst[0]):
        return ("Call "+lst[0][lst[0].find("$"):], True)
    if("RAISE" in lst[0]):
        return ("Raise "+lst[0][lst[0].find("$"):], True)
    return ''
    # lst = []
    # basewidth = 50
    # img_orig = img.crop(box = bx)
    # img_orig = ImageOps.invert(img_orig)
    
    # img_o = img_orig.load()

    # deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 10][0] for x in range(0, 9)])/255, sum([img_o[x, 10][1] for x in range(0, 9)])/255, sum([img_o[x, 10][2] for x in range(0, 9)])/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
    # #check to see if a bet is even on the table before processing an image
    # if (deltacolorC < 79):
    #     wpercent = (basewidth / float(img_orig.size[0]))
    #     hsize = int((float(img_orig.size[1]) * float(wpercent)))

    #     img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
    #     img_resized = binarize_array(img_resized, 130)
    #     #img_resized.show()

    #     #img_med = img_resized.filter(ImageFilter.MedianFilter)
    #     #img_sharp = img_resized.filter(ImageFilter.SHARPEN)

    #     lst.append(pytesseract.image_to_string(img_resized, 'Eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.$'))
    #     # if (lst[0] == ''):
    #     #     lst.append(pytesseract.image_to_string(img_sharp, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=Waitforbglnd'))
    #     #print(img_o)
    #     #print(deltacolorC)
    #     return lst[0][lst[0].find("$"):]
    # return ''

print(test((), img.crop(box = (495, 475, 590, 492))))


# toplist = []
# winlist = []

# def enum_cb(hwnd, result):
#     winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

# win32gui.EnumWindows(enum_cb, toplist)

# firefox = [(hwnd, title) for hwnd, title in winlist if '.02' in title.lower()]
# # just grab the hwnd for first window matching firefox
# firefox = firefox[0]
# hwnd = firefox[0]

# win32gui.SetForegroundWindow(hwnd)
# bbox = win32gui.GetWindowRect(hwnd)
# img = ImageGrab.grab(bbox)
# img.save("dealer 7.png")