# holdem hand evaluator from https://stackoverflow.com/questions/10363927/the-simplest-algorithm-for-poker-hand-evaluation


def holdem(board, hands):
    scores = [(evaluate((board + ' ' + hand).split()), i) for i, hand in enumerate(hands)]
    best = max(scores)[0]
    return [x[1] for x in filter(lambda x: x[0] == best, scores)]

def evaluate(hand):
    ranks = '23456789TJQKA'
    if len(hand) > 5: return max([evaluate(hand[:i] + hand[i+1:]) for i in range(len(hand))])
    score, ranks = zip(*sorted((cnt, rank) for rank, cnt in {ranks.find(r): ''.join(hand).count(r) for r, _ in hand}.items())[::-1])
    if len(score) == 5: # if there are 5 different ranks it could be a straight or a flush (or both)
        if ranks[0:2] == (12, 3): ranks = (3, 2, 1, 0, -1) # adjust if 5 high straight
        score = ([(1,),(3,1,2)],[(3,1,3),(5,)])[len({suit for _, suit in hand}) == 1][ranks[0] - ranks[4] == 4] # high card, straight, flush, straight flush
    return score, ranks

