"""
 * Use screengrab to parse different areas of the image for interpretation
 * 
"""
import screengrab
import time
import pytesseract
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from PIL import Image, ImageFilter, ImageOps

class interpreter():
    # coordinates
    Bigblind = (347, 465, 490, 503)
    iamback = (518, 475, 600, 495)
    totpotb4flop = (497, 198, 547, 214)
    totpot = (497, 188, 560, 204)
    mainpot = (492, 210, 545, 225)
    fold = (372, 475, 420, 492)
    call = (495, 475, 590, 492) #and check
    raze = (645, 475, 790, 492) #and bet and allin
    table = (345, 240, 390, 315)
    # card1, card2, button, equity, bet
    players = [
    [(450, 355, 480, 405), (490, 355, 520, 405), (525, 370, 540, 385), (465, 405, 535, 430), (470, 337, 530, 354)],
    [(175, 305, 205, 355), (210, 305, 240, 355), (275, 360, 290, 375), (190, 355, 260, 380), (267, 317, 327, 334)],
    [(175, 145, 205, 195), (210, 145, 240, 195), (275, 195, 290, 210), (190, 190, 260, 215), (288, 219, 348, 236)],
    [(450, 90, 480, 135), (490, 90, 515, 135), (550, 130, 565, 145), (465, 137, 535, 162), (568, 168, 618, 185)],
    [(725, 145, 755, 195), (760, 145, 790, 195), (685, 195, 700, 210), (740, 190, 810, 215), (665, 220, 715, 237)],
    [(725, 310, 755, 360), (760, 310, 790, 360), (685, 360, 700, 375), (740, 355, 810, 380), (690, 317, 740, 334)]]


    def __init__(self, logger):
        self.logger = logger
        self.s = screengrab.Screengrab()
        while(self.get_state(['', '', [], []])[0] == "Client not open"):
            time.sleep(5)
        
        
        

    def process_image(self, img_orig, vers):
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


        
         # check if cards are folded/present
        
        #nothing to process
        img_o = img_orig.load()

        # vers 1 is for the table cards and hand cards
        if vers == 1:
            #check to see if back of card or table
            
            deltacolorO = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 15][0] for x in range(5, 15)])/10/255, sum([img_o[x, 15][1] for x in range(5, 15)])/10/255,sum([img_o[x, 15][2] for x in range(5, 15)])/10/255), LabColor), convert_color(sRGBColor(1, 105/255, 0), LabColor))
            deltacolorG = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 25][0] for x in range(5, 15)])/10/255, sum([img_o[x, 25][1] for x in range(5, 15)])/10/255,sum([img_o[x, 25][2] for x in range(5, 15)])/10/255), LabColor), convert_color(sRGBColor(93/255, 93/255, 93/255), LabColor))
            #print(str(deltacolorO) + " - " + str(deltacolorG))
            #print(img_o[13, 35])
            
            if deltacolorO<=15 or deltacolorG<=15:
                #img_orig.show()
                return 'f'
            
            img_temp = img_orig
            img_orig = img_orig.crop(box = (0, 0, 35, 35))
            
            lst = []
            basewidth = 35
            wpercent = (basewidth / float(img_orig.size[0]))
            hsize = int((float(img_orig.size[1]) * float(wpercent)))
            img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
            img_resized = binarize_array(img_resized, 200)


            img_sharp = img_resized.filter(ImageFilter.SHARPEN)

            lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=23456789JQKA10'))
            if (lst[0] == ''):
                lst.append(pytesseract.image_to_string(img_sharp, 'eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=23456789JQKA10'))
            
            #fixes a bug where some spades cards can be interpreted as having a trailing zero
            if (len(lst[0])>=2 and not lst[0] == '10'):
                if (lst[0][0] in"23456789JQKA"):
                    lst[0] = lst[0][0]
        

            self.logger.debug(lst)
            
            while '' in lst:
                lst.remove('')
            
            
            
            # figure out the suit based on color
            if not lst == []:
                if lst[0] == '10':
                    lst[0] = "T"
                
                #print(deltacolorW)
                if img_temp.size[0] > 40:
                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 50][0] for x in range(20, 28)])/8/255, sum([img_o[x, 50][1] for x in range(20, 28)])/8/255,sum([img_o[x, 50][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(50/255, 160/255, 40/255), LabColor))
                    print(deltacolorC)
                    if deltacolorC < 25:
                        return lst[0] + "c"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 50][0] for x in range(20, 28)])/8/255, sum([img_o[x, 50][1] for x in range(20, 28)])/8/255, sum([img_o[x, 50][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(0, 0, 0), LabColor))
                    print('hi')
                    if deltacolorC < 30:
                        return lst[0] + "s"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 50][0] for x in range(20, 28)])/8/255, sum([img_o[x, 50][1] for x in range(20, 28)])/8/255, sum([img_o[x, 50][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(0, 138/255, 207/255), LabColor))
                    if deltacolorC < 20:
                        return lst[0] + "d"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 50][0] for x in range(20, 28)])/8/255, sum([img_o[x, 50][1] for x in range(20, 28)])/8/255, sum([img_o[x, 50][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(200/255, 0, 0), LabColor))
                    if deltacolorC < 20:
                        return lst[0] + "h"

                else:
                    #otherwise it is a handcard
                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 36][0] for x in range(20, 28)])/8/255, sum([img_o[x, 36][1] for x in range(20, 28)])/8/255, sum([img_o[x, 36][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(50/255, 160/255, 40/255), LabColor))
                    if deltacolorC < 25:
                        return lst[0] + "c"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 36][0] for x in range(20, 28)])/8/255, sum([img_o[x, 36][1] for x in range(20, 28)])/8/255, sum([img_o[x, 36][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(0, 0, 0), LabColor))
                    print(deltacolorC)
                    if deltacolorC < 35:
                        return lst[0] + "s"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 36][0] for x in range(20, 28)])/8/255, sum([img_o[x, 36][1] for x in range(20, 28)])/8/255, sum([img_o[x, 36][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(0, 136/255, 207/255), LabColor))
                    if deltacolorC < 25:
                        return lst[0] + "d"

                    deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 36][0] for x in range(20, 28)])/8/255, sum([img_o[x, 36][1] for x in range(20, 28)])/8/255, sum([img_o[x, 36][2] for x in range(20, 28)])/8/255), LabColor), convert_color(sRGBColor(200/255, 0, 0), LabColor))
                    if deltacolorC < 20:
                        return lst[0] + "h"
            #img_resized.show()
            return 'fail'

        # vers 2 is for the button
        if vers == 2:
            img_o = img_orig.load()
            deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 10][0] for x in range(0, 9)])/255, sum([img_o[x, 10][1] for x in range(0, 9)])/255, sum([img_o[x, 10][2] for x in range(0, 9)])/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
            if (deltacolorC > 75):
                # lst = []
                # basewidth = 15
                # wpercent = (basewidth / float(img_orig.size[0]))
                # hsize = int((float(img_orig.size[1]) * float(wpercent)))
                # img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
                # img_resized = binarize_array(img_resized, 90)

                # lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=D'))
                # return lst[0]
                return 'D'
            return ''
        
        # processing for "waiting for BB" image
        if vers == 3:
            lst = []
            basewidth = 125
            img_orig = img_orig.crop(box = (380, 465, 460, 500))
            img_orig = ImageOps.invert(img_orig)
            wpercent = (basewidth / float(img_orig.size[0]))
            hsize = int((float(img_orig.size[1]) * float(wpercent)))

            img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
            img_resized = binarize_array(img_resized, 65)

            lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=Waitforbglnd'))
            
            return lst[0]
        
        # processing for bet values ()   
        if vers == 4:
            lst = []
            basewidth = 50
            img_orig = ImageOps.invert(img_orig)
            
            img_o = img_orig.load()

            deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 10][0] for x in range(0, 9)])/255, sum([img_o[x, 10][1] for x in range(0, 9)])/255, sum([img_o[x, 10][2] for x in range(0, 9)])/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
            #check to see if a bet is even on the table before processing an image
            if (deltacolorC < 79):
                wpercent = (basewidth / float(img_orig.size[0]))
                hsize = int((float(img_orig.size[1]) * float(wpercent)))

                img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
                img_resized = binarize_array(img_resized, 130)
                #img_resized.show()

                #img_med = img_resized.filter(ImageFilter.MedianFilter)
                #img_sharp = img_resized.filter(ImageFilter.SHARPEN)

                lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.$'))
                # if (lst[0] == ''):
                #     lst.append(pytesseract.image_to_string(img_sharp, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=Waitforbglnd'))
                #print(img_o)
                #print(deltacolorC)
                if('$' in lst[0]):
                    return lst[0][lst[0].find("$"):]
                return lst[0]
            return ''
        
        #processing for fold, call, raze
        if vers == 5:
            lst = []
            basewidth = 125

            img_orig = ImageOps.invert(img_orig)
            wpercent = (basewidth / float(img_orig.size[0]))
            hsize = int((float(img_orig.size[1]) * float(wpercent)))
            
            img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
            img_resized = binarize_array(img_resized, 170)
            
            #img_resized.show()
            
            lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=BTCALRISEHKNFOD$0123456789.'))
            #print(lst[0])
            if("FOLD" in lst[0] or "FOL" in lst[0]):
                return("Fold", True)
            if("CALL" in lst[0]):
                return ("Call - "+lst[0][lst[0].find("$"):], True)
            if("CHECK" in lst[0]):
                return ("Check", True)
            if("RAISE" in lst[0]):
                return ("Raise - "+lst[0][lst[0].find("$"):], True)
            if("ALL" in lst[0] or "IN" in lst[0]):
                return("All-in", True)
            if("BET" in lst[0]):
                return("Bet - "+lst[0][lst[0].find("$"):], True)
            
            return '_'
        
        if vers == 6:
            lst = []
            basewidth = 50
            
            img_o = img_orig.load()
            wpercent = (basewidth / float(img_orig.size[0]))
            hsize = int((float(img_orig.size[1]) * float(wpercent)))

            img_resized = img_orig.convert('L').resize((basewidth, hsize), Image.ANTIALIAS)
            img_resized = binarize_array(img_resized, 150)
            #img_resized.show()
            #img_med = img_resized.filter(ImageFilter.MedianFilter)
            #img_sharp = img_resized.filter(ImageFilter.SHARPEN)

            lst.append(pytesseract.image_to_string(img_resized, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.$'))
            # if (lst[0] == ''):
            #     lst.append(pytesseract.image_to_string(img_sharp, 'eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=Waitforbglnd'))
            #print(img_o)
            #print(deltacolorC)
            if('$' in lst[0]):
                return lst[0][lst[0].find("$"):]
            return lst[0]
        
        if vers == 7:
            #img_orig.show()
            deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 500][0] for x in range(535, 544)])/9/255, sum([img_o[x, 500][1] for x in range(535, 544)])/9/255, sum([img_o[x, 500][2] for x in range(535, 544)])/9/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
            if(deltacolorC == 0):
                return "BACK"
            

    
    



    def get_state(self, prev_state):
        """ 
        Get the current state of the game
        input : nothing
        output: tuple containing the current state and a boolean representing the expectation of a client response 
        (state, expecting_response)
        """


        # 1: check if game client is open
        img = self.screenshot()
        
        if(img == ''):
            return ("Client not open.|", False, [], [])

        

        # 2: check if waiting for blind/just joined table
            # replace this with just checking if folded. They're essentially the same thing

        # val = self.process_image(img, 3)
        # if ('wait' in val or 'for' in val or 'blind' in val or 'big' in val):
        #     return ("Waiting for BB.|", False, [], [])

        # check if player is back
        bak = self.process_image(img, 7)
        if("BACK" == bak):
            return ("I Am Back|", False, [], [])

        pl0 = self.get_playern(0, img)


        # 3: check if folded ()
        if('f' in pl0 or 'fail' in pl0):
            print(pl0)
            return ("Folded. Waiting for next hand.|", False, [], [])

        # 3: preround betting - (cards in hand, no flop)


        # check if state changed by comparing table cards to pseudotable cards
        pseudotable = self.get_pseudo_card(img)
        table = prev_state[2] if len(pseudotable) == len(prev_state[2]) else self.get_table_cards(img)


        if('f' in table or 'fail' in table):
            #self.logger.error("Table card could not be read. Waiting 1 second and trying again")
            time.sleep(1)
            img = self.screenshot()
            table = self.get_table_cards(img)
            if('fail' in table or 'f' in table):
                return ("Aborting|", True, [], pl0, [])
        

        fld = self.process_image(img.crop(self.fold), 5) #fold
        cx = self.process_image(img.crop(self.call), 5)  #check and call 
        bet = self.process_image(img.crop(self.raze), 5) #bet raise and allin
        turn = fld[0] + '|' + cx[0] + '|' + bet[0]
        print(turn)
        if(len(table) == 0):
            pot = self.process_image(img.crop(self.totpotb4flop), 4)
            print(pot)
            if(turn == '_|_|_'):
                return ("Prebetting|", False, table, pl0, [pot])
            return ("Prebetting|"+turn, True, table, pl0, [pot])

        pot = self.process_image(img.crop(self.totpot), 4)
        mainpot = self.process_image(img.crop(self.mainpot), 4)
        # 4: flop - (three cards on table)
        if(len(table) == 3):
            if(turn == '_|_|_'):
                return ("Flop|", False, table, pl0, [pot, mainpot])
            return ('Flop|'+turn, True, table, pl0, [pot, mainpot])

        # 5: turn - (four cards on table)

        if(len(table) == 4):
            if(turn == '_|_|_'):
                return ("Turn|", False, table, pl0, [pot, mainpot])
            return ('Turn|'+turn, True, table, pl0, [pot, mainpot])

        # 6: river - (5 cards on table)
    
        if(len(table) == 5):
            if(turn == '_|_|_'):
                return ("River|", False, table, pl0, [pot, mainpot])
            return ('River|'+turn, True, table, pl0, [pot, mainpot])

        return("Aborting|", True, table, pl0, [pot, mainpot])

    def get_playern(self, n, img):
        playern = []
        # get card 1
        c = img.crop(box = self.players[n][0])
        playern.append(self.process_image(c, 1))
        # get card 2
        c = img.crop(box = self.players[n][1])
        playern.append(self.process_image(c, 1))
        # get button
        c = img.crop(box = self.players[n][2])
        playern.append(self.process_image(c, 2))
        # get bet amount
        c = img.crop(box = self.players[n][4])
        playern.append(self.process_image(c, 4))
        if(n == 0):
            # get starting equity
            c = img.crop(box = self.players[n][3])
            #c.show()
            playern.append(self.process_image(c, 6))
            print("---------------------------------------------------------")
            print(playern[-1])
            print("---------------------------------------------------------")
        return playern
    
    def get_players(self, img):
        # get the state of each player and return it as a string
        # card, card, button, bet, equity (if bot)
        players = [self.get_playern(x, img) for x in range(0,6)]

        for i in players:
            print(i[0])
            print(i[1])
            print(i[2])
            print(i[3])

        return players

    def get_table_cards(self, img):
        cards = []
        for i in range(5):
            t = (self.table[0] + 60*i, self.table[1], self.table[2]+ 60*i, self.table[3])
            c = img.crop(box = t)
            #c.show()
            cards.append(self.process_image(c, 1))
        while 'f' in cards:
            cards.remove('f')
        if ('fail' in cards):
            img.save(cards[0]+cards[1]+'.png')
        return cards

    def get_pseudo_card(self, img):
        cards = []

        for i in range(5):
            t = (self.table[0] + 60*i, self.table[1], self.table[2]+ 60*i, self.table[3])
            c = img.crop(box = t)
            img_o = c.load()
            deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 0][0] for x in range(0, 9)])/9/255, sum([img_o[x, 0][1] for x in range(0, 9)])/9/255, sum([img_o[x, 0][2] for x in range(0, 9)])/9/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
            if(deltacolorC == 0):
                cards.append(1)
                
        return cards

    def is_animating(self, img):
        # Given an image, check if it is animating by checking the colors of various card locations
        # Check Table 
        for i in range(5):
            t = (self.table[0] + 60*i, self.table[1], self.table[2]+ 60*i, self.table[3])
            c = img.crop(box = t)
            img_o = c.load()
            deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 0][0] for x in range(18, 27)])/9/255, sum([img_o[x, 0][1] for x in range(18, 27)])/9/255, sum([img_o[x, 0][2] for x in range(18, 27)])/9/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
            deltacolorG = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x, 0][0] for x in range(18, 27)])/9/255, sum([img_o[x, 0][1] for x in range(18, 27)])/9/255, sum([img_o[x, 0][2] for x in range(18, 27)])/9/255), LabColor), convert_color(sRGBColor(68/255, 68/255, 68/255), LabColor))
            
            if(not deltacolorC < 1 and not deltacolorG < 3.5):
                return True
            
        pl0 = self.players[0]
        img_o = img.crop(pl0[0]).load()
        deltacolorC = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x + 2, 2][0] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][1] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][2] for x in range(10, 19)])/9/255), LabColor), convert_color(sRGBColor(1, 1, 1), LabColor))
        deltacolorG = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x + 2, 2][0] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][1] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][2] for x in range(10, 19)])/9/255), LabColor), convert_color(sRGBColor(68/255, 68/255, 68/255), LabColor))    
        deltacolorF = delta_e_cie2000(convert_color(sRGBColor(sum([img_o[x + 2, 2][0] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][1] for x in range(10, 19)])/9/255, sum([img_o[x + 2, 2][2] for x in range(10, 19)])/9/255), LabColor), convert_color(sRGBColor(161/255, 161/255, 161/255), LabColor))
        if(not deltacolorC < 1 and not deltacolorG < 1 and not deltacolorF < 1):
            return True
        return False

                

    def screenshot(self):
        ss = self.s.screenshot()
        # if screenshot failed, tried to make foreground and try again
        
        while ss == '':
            ss = self.s.screenshot()

        if(self.is_animating(ss)):
            #ss.show()
            #print("The image is currently in an animation")
            self.screenshot()

        
        return ss
