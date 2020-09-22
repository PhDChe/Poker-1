import copy
import random

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
    def __init__(self, hand = [], community_cards=[], a_credits=0, b_credits=0, curr_pot_diff=0, pot=0,
                a_invested=0, b_invested=0):
        self.action_space = 6 #fold, check, call, raise12, raise1, allin
        self._max_episode_steps = 10000
        self.c2n = {'2' : 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        self.credits = [a_credits, b_credits]
        self.hand = hand[:]
        self.community_cards = community_cards[:]
        self.playerJustMoved = 1 # At the root, pretend the player just moved is 1 (opp). Player 0 (us) has first move
        self.pot = pot
        self.invested = [a_invested, b_invested]
        self.opp_hand = None
        # print('---------------------------------------------- ', curr_pot_diff)
        # self.moves_taken = [(1, 5), (1, 3)]
        # assume raise
        self.moves_taken = [(1, a_invested), (2, b_invested)]

        if self._get_pot_difference() == 0:
            #last person either called or checked; assume call. whatever the case we are in the next game state
            if(a_invested == 10):
                self.moves_taken = [(3, a_invested), (1, b_invested)]
            else:
                self.moves_taken = [(4, a_invested), (3, b_invested)]

        

        self.upState()
    
    def reset(self):
        self.__init__()
        return self.state

    def clone(self):
        ps = PokerState(self.hand, self.community_cards, self.credits[0],
                        self.credits[1], self._get_pot_difference(), self.pot, self.invested[0],
                        self.invested[1])

        ps.playerJustMoved = self.playerJustMoved
        ps.upState()
        return copy.deepcopy(self)
        #return ps

    def upState(self):
        self.state = {'obs': [0]*54}
        # print(self.hand + self.community_cards)
        for i in self.hand + self.community_cards:
            # print(i, [((i[0]-1)%13)+13*['s', 'h', 'd', 'c'].index(i[1])])
            self.state['obs'][((self.c2n[i[1]]-1)%13)+13*['S', 'H', 'D', 'C'].index(i[0])] = 1

        self.state['obs'][52] = self.invested[0] #(self.playerJustMoved+1)%2
        self.state['obs'][53] = self.invested[1]
        self.state['cur'] = ((self.playerJustMoved+1)%2)
        self.state['stage'] = self._get_stage_num()
        self.state['moves'] = self.get_moves()
        self.state['legal_actions'] = []
        
        self.state['state'] = (self.credits, self.hand, self.community_cards, self.playerJustMoved, self.pot, self.invested, self.opp_hand, self.get_moves() == [], self.moves_taken)

        for i in self.get_moves():
            self.state['legal_actions'].append(i)
        # print(self.state)
    
    def loadState(self, state):
        # fill local data with state data
        self.credits = state['state'][0]
        self.hand = state['state'][1]
        self.community_cards = state['state'][2]
        self.playerJustMoved = state['state'][3]
        self.pot = state['state'][4]
        self.invested = state['state'][5]
        self.opp_hand = state['state'][6]

        self.moves_taken = state['state'][8]


    def do_move(self, move):
        #self.moves_taken += [move]
        player = self._get_player_turn()
        diff = self._get_pot_difference()
        
        # print(move, diff)
        if(move == 0):
            self.moves_taken += [(0, 0)]
        elif move == 1 or move == 2:
            # if 2 is all in, make sure diff is only equal to his stack amt
            if (diff > self.credits[player]):
                self.pot += self.credits[player]
                self.moves_taken += [(move, self.credits[player])]
                self.invested[player] += self.credits[player]
                self.credits[player] = 0
            else:
                self.moves_taken += [(move, diff)]
                self.pot += diff
                self.invested[player] += diff
                self.credits[player] -= diff
        
        elif move == 3:
            self.credits[player] -= (self.pot//2)
            self.invested[player] += (self.pot//2)
            self.moves_taken += [(move, self.pot//2)]
            self.pot += (self.pot//2)
        elif move == 4:
            self.credits[player] -= (self.pot)
            self.invested[player] += (self.pot)
            self.moves_taken += [(move, self.pot)]
            self.pot += (self.pot)

        elif move == 5:
            self.moves_taken += [(move, self.credits[player])]
            self.pot += self.credits[player]
            self.invested[player] += self.credits[player]
            self.credits[player] = 0
            
        
        # print(self.moves_taken)

        
        self.playerJustMoved = (self.playerJustMoved + 1) % 2
        self.upState()
        return self.state, self.get_result(self.playerJustMoved)

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
        

        actions = [0, 1, 2, 3, 4, 5]


        if diff > 0:
            actions.remove(1)

        if diff == 0:
            actions.remove(2)
        
        if pot >= self.credits[player] or pot == diff:
            actions.remove(4)
        
        if pot // 2 >= self.credits[player] or pot//2 == diff:
            actions.remove(3)

        if diff > 0 and self.invested[player] + diff >= self.credits[player]:
            actions = [0, 2]
            if (self.credits[player] > 0 and self.credits[(player+1)%2] > 0):
                actions.append(5)

        return actions


    def get_result(self, playerjm):
        chips = 250
        if self._get_folded() == playerjm:
            return (chips-self.invested[playerjm])/chips # 0 # -log(self.invested[playerjm]) # 0 #-self.invested[playerjm] # return   # 0
        elif self._get_folded() == (playerjm + 1) % 2:
            return (self.pot + (chips - self.invested[playerjm]))/chips
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
            # if(player_2 == player_0):
            #     return (chips-self.invested[playerjm])/chips
            if player_0 == max(player_0, player_2):
                return (self.pot + (chips - self.invested[playerjm]))/chips if playerjm == 0 else (chips-self.invested[playerjm])/chips  #
            else:
                return (chips-self.invested[playerjm])/chips if playerjm == 0 else (self.pot + (chips - self.invested[playerjm]))/chips
    
    def to_dict(self):
        """
        return the current state as a dict with observation s 
        """

        lst = []
        
        
        return {'obs': lst}

    def _get_player_turn(self):
        return len(self.moves_taken) % 2

    def _get_folded(self):
        if len(self.moves_taken) == 0 or self.moves_taken[-1] != 0:
            return -1
        else:
            return (len(self.moves_taken) + 1) % 2

    def _get_pot_difference(self):
        return abs(self.invested[0] - self.invested[1])

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

