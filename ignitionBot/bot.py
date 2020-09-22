"""
 * Use all supplemental files to run a bot on Ignition casino
"""
import os
import logging
from action import action
from datetime import datetime
import time
import threading
import interpreter
import mouse
from treys import evaluator, Card
from holdem_calc import holdem_calc
from poker.hand import Combo, Range
from ivan import Ivan
#import holdem_calc


class ignitionBot():

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig()
        self.val = 0
        self.interpret = interpreter.interpreter(self.logger)   
        self.ev = evaluator.Evaluator()
        self.start = datetime.now()
        self.act = action()
        self.clear = lambda: os.system('cls')

        self.hands = []
        self.folded = 0
        self.won = 0
        self.pl = 0
        self.er = 0
        self.startVal = 120

        self.positioning = ['BTN','CO', 'HJ', 'UTH', 'BB', 'SB']
        self.early_range = Range('ATs+, KTs+, 77+, QTs+, T9s, KJ+, ATs+')
        self.mid_range  = Range('55+, A6s+, K9s+, Q8s+, J8s+, 98s+, KT+, QT+, JT+')
        self.late_range = Range('22+, A2s+, K2s+, Q8s+, J7s+, 98, 87, 76, 65, 54')
        self.allin_range = Range('QQ+, AK+')

        self.state = ["|", "", [], []]
        self.prevstate = ["|", "", [], []]
        self.position = None
        self.winp = ""
        self.losep = ""
        self.tiep = ""
        self.percent = ""
        self.hnd = ""
        self.notes = ""
        self.pos = ""
        self.board = []
        self.handcards = [] 

        os.system('SET OMP_THREAD_LIMIT=1')
        self.output_loop()
        self.run()


    def output_loop(self):
        threading.Timer(1, self.output_loop).start()
        

        output = ''
        output = output + self.state[0][:self.state[0].find("|")].center(80, ' ')+'\n'+"================================================================================\n"
        deltatime = datetime.now() - self.start
        output = output + Card.print_pretty_cards(self.board).center(38, ' ')+"||"+deltatime.__str__().center(38, ' ')+'\n'
        output = output + Card.print_pretty_cards(self.handcards).center(38, ' ')+"||"+"hands played"+str(len(self.hands)).center(25-len(self.hands)//10, " ")+"\n"
        output = output + "hand %"+"".center(32, ' ')+"||"+ "won         "+str(self.won).center(25-self.won//10, ' ')+"\n"
        output = output + "win  %"+self.winp.center(32, ' ')+"||"+"folded      "+str(self.folded).center(25-self.folded//10, ' ')+"\n"
        output = output + "lose %"+self.losep.center(32, ' ')+"||"+"P/L         "+str(self.pl).center(25, ' ')+"\n"
        output = output + "tie  %"+self.tiep.center(32, ' ')+ '||'+''.join(self.notes).center(38,' ')+'\n'
        output = output + "errors"+str(self.er).center(32, ' ')
        #self.clear()
        print(output, end = '\r')


    def run(self):
        while 1:
        
            # take screenshot
            # get state of program
            try:
                self.state = self.interpret.get_state(self.state)
            except Exception as e:
                self.er += 1
                print(e)
            
            if(not "Client not open.|" == self.state[0]):
                try:
                    self.position = self.interpret.s.position()
                    #self.act = action()
                    self.game()
                except Exception as e:
                    self.er += 1
                    print(e)
            else:
                time.sleep(1)
            

    def game(self):
        tablecards = self.state[2]
        if('Folded' in self.state[0] or 'Aborting' in self.state[0] or 'Waiting for BB' in self.state[0]):
            img = self.interpret.screenshot()
            #tablecards = self.interpret.get_table_cards(img)
            while('f' in tablecards):
                tablecards.remove('f')
            while('fail' in tablecards):
                tablecards.remove('fail')
            #print(tablecards)
            self.handcards = []
            self.board = [Card.new(x) for x in tablecards]
        elif(not self.state == self.prevstate and not 'Back' in self.state[0]):
            # 'Fold' 'Call/Check
            self.board = [Card.new(x) for x in tablecards]

            locked = False
            hand = [self.state[3][0], self.state[3][1]]
            if ('f' in hand or 'fail' in hand):
                locked = True
            else:
                com = Combo(hand[0]+hand[1])

            simulationEarly = holdem_calc.calculate_odds_villan(tablecards, False, 30, None, com, None, False, True)
            self.winp = str(simulationEarly[0]['win'])
            self.tiep = str(simulationEarly[0]['tie'])
            self.losep = str(simulationEarly[0]['lose'])
            # simulationMid = holdem_calc.calculate_odds_villan(tablecards, False, 50, None,  com, mid_range, False, False)
            # simulationLate = holdem_calc.calculate_odds_villan(tablecards, False, 50, None, com, late_range, False, False)
            # winp = winp + str(sum([simulationLate[0]['win'], simulationMid[0]['win'], simulationEarly[0]['win']]))
            # tiep = tiep + str(sum([simulationLate[0]['tie'], simulationMid[0]['tie'], simulationEarly[0]['tie']]))
            # losep = losep + str(sum([simulationLate[0]['lose'], simulationMid[0]['lose'], simulationEarly[0]['lose']]))
            #percent = percent + str(ev.get_five_card_rank_percentage(hr))
            #hnd = hnd + str(ev.class_to_string(ev.get_rank_class(hr)))
            if(not locked and 'Prebetting' in self.state[0]):
                # u = self.interpret.get_playern(0, img)
                if ('f' not in hand and 'fail' not in hand):
                    self.handcards = [Card.new(hand[0]), Card.new(hand[1])]

            if(self.state[1] == True and not locked):
                #possible moves
                moves = self.state[0].split("|")[1:]
                img = self.interpret.screenshot()
                players = self.interpret.get_players(img)

                c = 0
                for x in players:
                    if x[2] == "D":
                        break
                    c += 1

                self.pos = self.positioning[c%6]
                
    


                if('Prebetting' in self.state[0]):
                    # add hand to history
                    self.hands.append(com)
                    action = ""
                    # call/raise; check/bet
                    if(com in self.allin_range.combos):
                        action = 'allin'

                    elif(com in self.early_range.combos):
                        action = 'call/raise'                        
                    
                    elif(com in self.mid_range.combos):
                        action = 'call/raise'
                        if c > 4:
                            action = 'check/fold'
                    
                    elif(com in self.late_range.combos):
                        action = 'call/raise'
                        if c > 2:
                            action = 'check/fold'
                    else:
                        action = 'check/fold'
                    # check if hand is in correct range given current position



                elif('Flop' in self.state[0] or 'Turn' in self.state[0]):
                    self.hr = self.ev.evaluate(self.handcards, self.board)
                    self.percent = str(self.ev.get_five_card_rank_percentage(self.hr))
                    self.hnd = str(self.ev.class_to_string(self.ev.get_rank_class(self.hr)))

                    if(float(self.winp) > .7):
                        action = 'call/raise'
                    
                    elif(float(self.winp) > .6):
                        action = 'check/call'
                    
                    else:
                        action = 'check/fold'
                

                elif('River' in self.state[0]):
                    self.hr = self.ev.evaluate(self.handcards, self.board)
                    self.percent = str(self.ev.get_five_card_rank_percentage(self.hr))
                    self.hnd = str(self.ev.class_to_string(self.ev.get_rank_class(self.hr)))

                    if(float(self.winp) > .88):
                        action = 'allin'
                   
                    elif(float(self.winp) > .7):
                        action = 'call/raise'
                    
                    elif(float(self.winp) > .6):
                        action = 'check/call'
                    
                    else:
                        action = 'check/fold'
                    


                self.notes = action
                print(moves)                
                
                if('check/fold' == action):
                    if("Call" in moves[1]or 'All' in moves[2]):
                        self.act.fold_action()
                    else:
                        self.act.call_action()

                elif('allin' == action):
                    self.act.raise_action()
                    # if("Call" in moves[1]):
                    #     self.act.raise_action()
                    
                
                elif('call/raise' == action):
                    self.act.call_action()
                    if("Call" in moves[1]):
                        self.act.call_action()
                    else:
                        self.act.raise_action()
                
                                
                else:
                    self.act.fold_action()
                
                
                



            elif(self.state[1] == True and locked):
                moves = self.state[0].split("|")[1:]
                img = self.interpret.screenshot()
                players = self.interpret.get_players(img)
                if('Fold' in moves):
                    self.act.fold_action()
                if('Check' in moves):
                    self.act.call_action()
                #fold/check
                
            
            self.prevstate = self.state
            #get pl0's equity
            print(self.state[3])
            print(self.state)
            self.val = float(self.state[3][4])
            self.pl = self.val - self.startVal
        else:
            if("Back" in self.state[0]):
                self.act.i_am_back()



bot = ignitionBot()
