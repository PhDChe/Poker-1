"""
    Creates n number of decks
    and provides basic dealer functionality
"""
from Card import Card as card
from random import shuffle
class Deck:
    vals = "23456789TJQKA"
    def __init__(self, num):
        self.num = num
        self.deck = self.generate(self.num)

    
    def generate(self, num):
        d = []
        for i in range(num):
            for j in range(14):
                d.append(card(self.vals[j], 'S'))
                d.append(card(self.vals[j], 'C'))
                d.append(card(self.vals[j], 'D'))
                d.append(card(self.vals[j], 'H'))
        return d

    def shuffle(self):
        # shuffle the deck
        shuffle(self.deck)

    def draw(self):
        # remove and return the card at the top of the deck
        return self.deck.pop()

    def reset(self):
        # reset the deck with a new, shuffled deck
        self.deck = self.generate(self.num)
        self.shuffle()