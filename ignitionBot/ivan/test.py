"""
https://github.com/sozos/MCTS/blob/master/source/agents/mcts/node.py
"""
from math import sqrt, log
import random

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.get_moves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + 3*sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


"""
https://github.com/sozos/MCTS/blob/master/source/agents/mcts/uct.py
"""

def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.do_move(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.do_move(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.get_moves() != []: # while state is non-terminal
            # print('---------')
            # print(state.credits)
            # print(state._get_player_turn())
            # print(state.get_moves())
            # print(state.moves_taken)
            # probs = [1 for x in state.get_moves()]
            # if(5 in state.get_moves()):
            #     probs[-1] -= .5
            state.do_move(random.choice(state.get_moves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.get_result(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString()) 
    
    # determine general performance of hand
    avg = [x.wins/x.visits for x in rootnode.childNodes]
    avg =  sum(avg)/len(avg)


    ret = sorted(rootnode.childNodes, key = lambda c: c.visits)
    print(ret)
    # all responses good
    if(avg > .7):
        
        return ret[-1].move # return the move that was most visited
    
    for i in range(1, len(ret)+1):

        if(ret[-i].move[0] in [0, 1, 2, 3, 4]):
            return ret[i].move
    
    

HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH = range(9)
class Deck(object):
    def __init__(self, num_decks=1):
        """
        Generate a list of cards for num_decks decks, 52 cards per deck.
    
        codes:
            rank:
                2-10
                11 jack
                12 queen
                13 king
                14 ace
    
            suit:
                (d)iamonds
                (s)pades
                (h)earts
                (c)lubs
            
        Example Deck:
            [(2, 'd'), (4, 's'), ...]
        """
        self.all_cards = []
        for deck in range(num_decks):
            for suit in 'dshc':
                for value in range(2,15):
                    self.all_cards.append((value, suit))
        self.shuffle()
    
    def shuffle(self):
        self.cards = self.all_cards[:]
        random.shuffle(self.cards)
        
    def take(self, n):
        taken = self.cards[:n]
        self.cards = self.cards[n:]
        return taken
        
    def take_one(self):
        taken = self.take(1)
        return taken[0]
                
def is_flush(hand):
    _rank, flush_suit = hand[0]
    for card in hand[1:]:
        _rank, suit = card
        if suit != flush_suit:
            return False
    else:
        return True
        
def is_straight(sorted_hand):
    last_rank, _suit = sorted_hand[0]
    for i, card in enumerate(sorted_hand[1:]):
        rank, _suit = card
        # handle the case of a straight with an ace
        # acting as 1, though its rank is 14:
        # 14, 5, 4, 3, 2
        #     0, 1, 2, 3
        if rank == 5 and i == 0 and last_rank == 14:
            pass
        elif rank != last_rank - 1:
            return False
        last_rank = rank
    else:
        return True
        
def hand_rank(hand):
    """
    Convert a 5 card hand into a tuple representing its relative rank among
    all possible hands. While the magnitude is meaningless, the ordering is
    not, and the tuple tells you if one hand is worth more than another.
    
    This takes advantage of the properties of tuples in Python.  When you
    compare two tuples, the values in the tuple are compared in order until
    one is found that is different from the other, just like comparing strings.
    
    The first number in the tuple is the overall rank of the hand, the other
    numbers are only useful for comparing two hands of the same overall rank,
    for example, two two-pair hands are compared first on the rank of their
    respective pairs, then, if those are equal, on the other cards in the hand.
    This function is not particularly efficient.
    
    Example:
    
        >>> high_card = [(14, 'd'), (10, 'd'), (9, 's'), (5, 'c'), (4, 'c')]
        >>> hand_rank(high_card)
        (0, 14, 10, 9, 5, 4)
        >>> one_pair = [(10, 'c'), (10, 's'), (6, 's'), (4, 'h'), (2, 'h')]
        >>> hand_rank(one_pair)
        (1, 10, 6, 4, 2)
        >>> two_pair = [(13, 'h'), (13, 'd'), (2, 's'), (2, 'd'), (11, 'h')]
        >>> hand_rank(two_pair)
        (2, 13, 2, 11)
        >>> three_of_a_kind = [(8, 's'), (8, 'h'), (8, 'd'), (5, 's'), (3, 'c')]
        >>> hand_rank(three_of_a_kind)
        (3, 8, 5, 3)
        >>> straight = [(8, 's'), (7, 's'), (6, 'h'), (5, 'h'), (4, 's')]
        >>> hand_rank(straight)
        (4, 8)
        >>> flush = [(14, 'h'), (12, 'h'), (10, 'h'), (5, 'h'), (3, 'h')]
        >>> hand_rank(flush)
        (5, 14, 12, 10, 5, 3)
        >>> full_house = [(10, 's'), (10, 'h'), (10, 'd'), (4, 's'), (4, 'd')]
        >>> hand_rank(full_house)
        (6, 10, 4)
        >>> four_of_a_kind = [(10, 'h'), (10, 'd'), (10, 'h'), (10, 's'), (5, 'd')]
        >>> hand_rank(four_of_a_kind)
        (7, 10, 5)
        >>> straight_flush = [(7, 'h'), (6, 'h'), (5, 'h'), (4, 'h'), (3, 'h')]
        >>> hand_rank(straight_flush)
        (8, 7)
    """
    cards = list(reversed(sorted(hand)))
    assert(len(cards) == 5)
    
    # straights/flushes
    straight = is_straight(cards)
    flush = is_flush(cards)
    if straight and flush:
        return (STRAIGHT_FLUSH, cards[0][0])
    elif flush:
        return tuple([FLUSH] + [rank for rank, suit in cards])
    elif straight:
        return (STRAIGHT, cards[0][0])

    # of_a_kind
    histogram = {}
    for card in cards:
        rank, _suit = card
        histogram.setdefault(rank, [])
        histogram[rank].append(card)
    
    of_a_kind = {}
    for rank, cards in reversed(sorted(histogram.items())):
        num = len(cards)
        of_a_kind.setdefault(num, [])
        of_a_kind[num].append(rank)
    
    num = max(of_a_kind)
    ranks = of_a_kind[num]
    if num == 4:
        result = [FOUR_OF_A_KIND]
    elif num == 3:
        if 2 in of_a_kind:
            result = [FULL_HOUSE]
        else:
            result = [THREE_OF_A_KIND]
    elif num == 2:
        if len(ranks) == 1:
            result = [ONE_PAIR]
        if len(ranks) == 2:
            result = [TWO_PAIR]
    elif num == 1:
        result = [HIGH_CARD]
    else:
        raise Exception("Failed to evaluate hand rank")

    result += ranks
    # get the rest of the cards to complete the tuple
    for n in range(num-1, 0, -1):
        if n in of_a_kind:
            result += of_a_kind[n]
    
    return tuple(result)
    
def n_card_rank(cards):
    """
    Find the highest rank for a set of n cards.  Simply ranks all 5 card hands
    in the 7 cards and returns the highest.
    
    >>> n_card_rank([(10, 'h'), (10, 'd'), (10, 'h'), (10, 's'), (5, 'd'), (4, 'd'), (3, 'd')])
    (7, 10, 5)
    >>> n_card_rank([(10, 'h'), (10, 'd'), (10, 'h'), (6, 's'), (5, 'd'), (4, 'd'), (3, 'd')])
    (3, 10, 6, 5)
    >>> n_card_rank([(10, 'h'), (10, 'd'), (10, 'h'), (6, 's'), (5, 'd'), (4, 'd'), (4, 'd')])
    (6, 10, 4)
    >>> n_card_rank([(10, 'h'), (10, 'd'), (10, 'h'), (6, 'd'), (5, 'd'), (4, 'd'), (3, 'd')])
    (5, 10, 6, 5, 4, 3)
    """
    cards = list(reversed(sorted(cards)))
    return max(hand_rank(hand) for hand in choose(5, cards))
    
def choose(n, seq):
    """
    Return all n-element combinations of elements from seq
    
    >>> len(choose(5, [1,2,3,4,5,6,7]))
    21
    """
    if n == 1:
        return [[x] for x in seq]
    if len(seq) <= n:
        return [seq]
    subseq = seq[:]
    elem = subseq.pop()
    return [[elem] + comb for comb in choose(n-1, subseq)] + choose(n, subseq)
    

cardsDict = [(r, s) for s in ['d', 'c', 'h', 's'] for r in range(2, 15)]


class PokerState:
    def __init__(self, hand, community_cards, a_credits, b_credits, curr_pot_diff, pot,
                a_invested, b_invested):
        self.credits = [a_credits, b_credits]
        self.hand = hand[:]
        self.community_cards = community_cards[:]
        self.playerJustMoved = 1 # At the root, pretend the player just moved is 1 (opp). Player 0 (us) has first move
        self.pot = pot
        self.invested = [a_invested, b_invested]
        self.opp_hand = None

        # print('---------------------------------------------- ', curr_pot_diff)
        # self.moves_taken = [(1, 5), (1, 3)]
        self.moves_taken = [(1, a_invested), (1, b_invested)]
        if curr_pot_diff == 0:
            self.moves_taken = [(1, a_invested), (1, b_invested)]
        elif curr_pot_diff == 9:
            self.moves_taken = [(1, a_invested), (3, b_invested)]
        elif curr_pot_diff > 3:
            self.moves_taken = [(1, a_invested), (4, b_invested)]


        if b_credits == 0 and a_invested > 0:
            self.moves_taken = [(3, a_invested), (5, b_invested)]
        

    def clone(self):
        ps = PokerState(self.hand, self.community_cards, self.credits[0],
                        self.credits[1], self._get_pot_difference(), self.pot, self.invested[0],
                        self.invested[1])

        ps.playerJustMoved = self.playerJustMoved
        return ps

    def do_move(self, move):
        #self.moves_taken += [move]
        player = self._get_player_turn()
        diff = self._get_pot_difference()
        pot = self.pot
        # print(move, diff)
        if move[0] == 1 or move[0] == 2:
            # if 2 is all in, make sure diff is only equal to his stack amt
            if (diff > self.credits[player]):
                self.pot += self.credits[player]
                self.moves_taken += [(move[0], self.credits[player])]
                self.invested[player] += self.credits[player]
                self.credits[player] = 0
            else:
                self.pot += diff
                self.credits[player] -= diff
                self.moves_taken += [(move[0], diff)]
                self.invested[player] += diff
        
        elif move[0] == 3:
            self.credits[player] -= (pot//2)
            self.invested[player] += (pot//2)
            self.moves_taken += [(move[0], pot//2)]
            self.pot += (pot//2)
        elif move[0] == 4:
            self.credits[player] -= (pot)
            self.invested[player] += (pot)
            self.moves_taken += [(move[0], pot)]
            self.pot += (pot)

        elif move[0] == 5:
            self.moves_taken += [(move[0], self.credits[player])]
            self.pot += self.credits[player]
            self.invested[player] += self.credits[player]
            self.credits[player] = 0
            
        

        if(self.credits[player] < 0):
            print('lmfaoooo')
        
        self.playerJustMoved = (self.playerJustMoved + 1) % 2

    def get_moves(self):
        if self._get_folded() != -1 or self._get_stage_num() == 5 or self.credits[self._get_player_turn()] == 0:
            return []
        # if self.playerJustMoved == 0:
        #     return [4]
        # add more moves, like all in 
        player = self._get_player_turn()
        diff = self._get_pot_difference()
        pot = self.pot

        # action_names = ['fold', 'check', 'call', 'raise half-pot', 'raise pot', 'all-in']
        
        moves = [(0, 0)]
        if diff == 0:
            moves += [(1, 0)]
        
        #cant both check and call in the same move
        if (1, 0) not in moves:
            moves += [(2, diff)]
        # print(self.credits)
        # print(diff)
        if self.credits[(player + 1)%2] <= 0:
            return moves

        # 248 for first round. pot bigger after each other round tho
        if self.credits[player] > pot//2 and pot//2 >= diff:
            moves += [(3, pot//2)]

        if self.credits[player] > pot and pot > diff:
            moves += [(4, pot)]
        

        moves += [(5, self.credits[player])]
    

        return moves

    # def get_result(self, playerjm):
    #     if self._get_folded() == playerjm:
    #         return -self.invested[playerjm]  # 0
    #     elif self._get_folded() == (playerjm + 1) % 2:
    #         return self.pot - self.invested[playerjm]
    #     else:
    #         # evaluate
    #         if self.opp_hand is None:
    #             self.opp_hand = []
    #             while len(self.opp_hand) < 2:
    #                 c = cardsDict[random.randrange(52)]
    #                 if not c in self.hand + self.community_cards + self.opp_hand:
    #                     self.opp_hand += [c]
    #         while len(self.community_cards) < 5:
    #             c = cardsDict[random.randrange(52)]
    #             if not c in self.hand + self.community_cards + self.opp_hand:
    #                 self.community_cards += [c]
    #
    #         player_0 = n_card_rank(self.hand + self.community_cards)
    #         player_2 = n_card_rank(self.opp_hand + self.community_cards)
    #         if player_0 == max(player_0, player_2):
    #             return self.pot - self.invested[playerjm] if playerjm == 0 else -self.invested[playerjm]  #
    #         else:
    #             # return -self.invested[playerjm] if playerjm == 0 else self.pot - self.invested[playerjm]
    def get_result(self, playerjm):
        total_chips = 250
        if self._get_folded() == playerjm:
            return 0  # -self.invested[playerjm]  # 0
        elif self._get_folded() == (playerjm + 1) % 2:
            return (self.pot - self.invested[playerjm]) / total_chips
        else:
            # evaluate
            if self.opp_hand is None:
                self.opp_hand = []
                while len(self.opp_hand) < 2:
                    c = cardsDict[random.randrange(52)]
                    if not c in self.hand + self.community_cards + self.opp_hand:
                        
                        self.opp_hand += [c]
            while len(self.community_cards) < 5:
                c = cardsDict[random.randrange(52)]
                if not c in self.hand + self.community_cards + self.opp_hand:
                    self.community_cards += [c]

            player_0 = n_card_rank(self.hand + self.community_cards)
            player_2 = n_card_rank(self.opp_hand + self.community_cards)
            if player_0 == max(player_0, player_2):
                return (self.pot - self.invested[playerjm]) / total_chips if playerjm == 0 else 0  # -self.invested[playerjm]  #
            else:
                return 0 if playerjm == 0 else (self.pot - self.invested[playerjm]) / total_chips

    def _get_player_turn(self):
        return len(self.moves_taken) % 2

    def _get_folded(self):
        if len(self.moves_taken) == 0 or self.moves_taken[-1][0] != 0:
            return -1
        else:
            return (len(self.moves_taken) + 1) % 2

    def _get_pot_difference(self):
        sums = [0, 0]

        for i in range(1, len(self.moves_taken) + 1):
            sums[-i % 2] += self.moves_taken[-i][1]
        return abs(sums[0] - sums[1])

    def _get_stage_num(self):
        num = 0
        consecutive_check = 0

        for move in self.moves_taken:
            if move == 1:
                if consecutive_check == 1:
                    num += 1
                    consecutive_check = 0
                else:
                    consecutive_check = 1
            else:
                consecutive_check = 0
                if move == 2:
                    num += 1

        return num



# st = PokerState([(13, 's'), (2, 'c')], [(4, 'h'), (14, 'd'), (4, 'd'), (13, 'd')], 240, 220, 20, 40, 10, 30)
# m = UCT(st, 577, False)

# print(m)
# m = UCT(st, 677, False)

# print(m)
# m = UCT(st, 877, False)

# print(m)