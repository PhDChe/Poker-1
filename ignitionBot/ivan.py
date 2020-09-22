
from treys import evaluator, Card
from holdem_calc import holdem_calc
from poker.hand import Combo, Range
import numpy as np
import random





class Ivan():

    def __init__(self):
        
        self.positioning = ['BTN','CO', 'HJ', 'UTH', 'BB', 'SB']
        self.early_range = Range('ATs+, KTs+, 77+, QTs+, T9s+, KJ+')
        self.mid_range  = Range('55+, ATs+, K9s+, Q9s+, JTs+, KT+, QT+, JT+')
        self.late_range = Range('22+, A2s+, A8o+, K9s+, Q9s+, J9s+, JTo+, QTo+, KTo+, 98s, 87s, 76s, 65s, 54s')
        self.any_range = Range('22+, AJ+, KQ+, KJs, QJs+, ATs+')
        self.allin_range = Range('QQ+, AK+')
            
    def calc_odds_range(self, board, hero, rng):
        itms = [holdem_calc.calculate_odds_villan(board, False, 50, False, hero, x, False, True) for x in rng.combos]
        # print(itms)
        return {'win' : np.mean([r[0]['win'] for r in itms if r]), 'tie':np.mean([r[0]['tie'] for r in itms if r]), 'lose' : np.mean([r[0]['lose'] for r in itms if not r == None])}
        # return [odds.update({o: np.mean([r[0][o] for r in itms if r])}) for o in ['tie', 'win', 'lose']]
    
    def next_move(self, board, players, pot1, pot2):
        """
        INPUTS:
        board: list of strings representing table cards
        players: list of list objects where each list is a player, [card1, card2, button ("D" if they're under the button), bet amt, equity (if pl0)]
        pot1: totpot - the cumulative sum of the current hand's mainpots
        pot2: mainpot - the cumulative sum of the current bets
        """
        

        com = Combo(players[0][0] + players[0][1])
        print(com)
        simulationEarly = holdem_calc.calculate_odds_villan(board, False, 30, None, com, None, False, False)
        #holdem_calc.calculate_odds_villan(board, False, 30, None, com, ['', '', '', '?'])

        winp = str(simulationEarly[0]['win'])
        tiep = str(simulationEarly[0]['tie'])
        losep = str(simulationEarly[0]['lose'])

        print(winp)
        print(losep)
        print(tiep)

        # prebetting
        if(len(board) == 0):

            pass
        
        # flop
        if(len(board) == 3):
            pass

        # turn
        if(len(board) == 4):
            pass


        # river
        if(len(board) == 5):
            pass
        return






a = Ivan()
a.next_move(['As', 'Ad', '3s'], [['Ks', 'Kd']], [], [])

# simulationEarly = calc_odds_range(tablecards, Hero, Villan)
# # holdem_calc.calculate_odds_villan(tablecards, False, 50, None, Hero, Villan, False, True)

# winp = str(simulationEarly['win'])
# tiep = str(simulationEarly['tie'])
# losep = str(simulationEarly['lose'])

# print(winp)