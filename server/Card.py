"""
    simple card representation
"""

# values - 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (J), 12 (Q), 13 (K), 14 (A)
# suit   - c, s, d, h
class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
    
    def getValue(self):
        return self.value
    
    def getSuit(self):
        return self.suit

    def getCard(self):
        return self.getValue() + "" + self.getSuit()
