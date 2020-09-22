import time
import random
import pokerhands
from operator import attrgetter
import time
import pokerstrat
import server
from threading import Thread
from websocket_server import WebsocketServer




#card class
class Card:

    RANKS=['2','3','4','5','6','7','8','9','10','J', 'Q', 'K', 'A']

    SUITS=['h', 'c', 's', 'd']

    def __init__(self,rank, suit, faceup=True):

        self.rank=rank
        self.suit=suit
        self.values=[]
        self.__value=(Card.RANKS.index(self.rank)+1)
        
        self.faceup=faceup

    def __str__(self):

        if self.faceup:
            
            return str(self.rank)+str(self.suit)
        else:
            return 'XX'

    @property

    def value(self):

        v=self.__value

        return v

#hand class (also used for Player)

class Hand:

    serial=0

    def __init__(self, name, table, server, strategy='Random'):

        
        self.strategy=[]
        self.stratname=strategy
        strategy_class=getattr(pokerstrat, strategy)
        strat=strategy_class(self)
        self.strategy.append(strat)
            
        self.server = server
        
        self.cards=[]
        self.total_cards=(self.cards+table.cards)
        table.players.append(self)
        self.name=name
        
        Hand.serial+=1
        self.position=Hand.serial
        self.small_blind=False
        self.big_blind=False
        self.dealer=False
        self.hand_value=0
        self.rep=''
        self.tie_break=0
        self.raw_data=0
        self.is_folded=False
        self.stack=1000
        
        self.stake=0
        self.in_pot=0
        self.to_play=0
        self.all_in=False
        self.first_all_in=False
        self.raised=0
        self.carry_over=0

        #data points for play analysis:

        self.history=[]

        self.pots_played=0
        self.win=0
        self.raises=0
        self.calls=0
        self.checks=0

    @property

    def play_analysis(self):

        pass

    @property

    def get_position(self):

        return self.position%pot.table_size
    
    def __str__(self):


        rep='\n'+str(self.name)+'\t    stack='+str(self.stack)+'\n'
        
        if self.small_blind:
            rep+=' small blind'
        elif self.big_blind:
            rep+=' big blind'
        elif self.dealer:
            rep+=' dealer'
        

        return rep
   
        
    
    def get_value(self):
        
        self.total_cards=(self.cards+table.cards)
        
        rep, hand_value, tie_break, raw_data=pokerhands.evaluate_hand(self.total_cards)
        
        self.rep=str(rep)
        self.hand_value=hand_value
        self.tie_break=tie_break
        self.raw_data=raw_data
        
    
        return hand_value, rep, tie_break, raw_data

   
    def print_cards(self):

        rep=''

        if self.is_folded:
            rep='FF'

        else:

            for card in self.cards:

                rep+=str(card)+'  '
        

        print (rep)
        
        
    def flip(self):
            
        for card in self.cards: card.faceup=not card.faceup

    def fold(self, pot):

        self.is_folded=True
        self.in_pot=0
        self.stake=0
        self.raised=0
        
     
        print (str(self.name)+' folds')

        pot.folded_players.append(self)
        if self in pot.active_players:
        
            pot.active_players.remove(self)
        
                
        if pot.one_remaining:
            
            pot.stage=5

    def no_play(self, pot):
        
        next_player(pot)

        self.stake=0
        
       
    def check_call(self, pot):
        
        
        
        if self.to_play==0:
            print (str(self.name)+' checks')
        else:
                if self.to_play>self.stack:
                    self.stake=self.stack
                else:
                    self.stake=self.to_play
                print (str(self.name)+' calls '+str(self.stake))
                if pot.stage==0 and pot.raised==False:
                    pot.limpers+=1

        next_player(pot)
    
    
    def bet(self, pot, stake):
        
        if pot.already_bet:
            print (str(self.name)+' raises '+str(stake-self.to_play))
            self.raised+=1
            pot.limpers=0
            pot.raised=True
        else:
            print (str(self.name)+' bets '+str(stake))
        
            pot.already_bet=True
      
        self.stake=stake
        pots[-1].to_play+=(self.stake-self.to_play)
        
        
        
        
        next_player(pot, True)
        
    def ante(self, pot):
        
        if self.small_blind:
            self.stack-=BLINDS[0]
            pot.total+=BLINDS[0]
            self.in_pot+=BLINDS[0]
            
        if self.big_blind:
            self.stack-=BLINDS[1]
            pot.total+=BLINDS[1]
            pot.to_play=BLINDS[1]
            self.in_pot+=BLINDS[1]
        
                    
    def bust(self):

        print (str(self.name)+' is bust')
        list_index=table.players.index(self)
        for p in table.players[list_index+1:]:
            p.position-=1
            
        table.players.remove(self)
        
        
    def clear(self):

      self.cards=[]
      self.is_folded=False
      self.all_in=False
      self.raised=0
      

    def add(self, cards):

      self.cards.append(cards)

#__________represents the card deck - shuffled each round            
        
class Deck(Hand):

    def __init__(self):

        self.cards=[]

    def populate(self):

        for rank in Card.RANKS:

            for suit in Card.SUITS:

                card=Card(rank, suit)
                self.cards.append(card)

    def shuffle(self):

        random.shuffle(self.cards)

    def print_cards(self):

        rep=''

        for card in self.cards:

            rep+=str(card)+' '

        print (rep)

    def deal_to(self, hand, cards=1, faceup=True):

        if len(self.cards)<cards:
                print ('not enough cards to deal')
                
        elif len(self.cards)==0:
                print ('deck empty')
                
        else:
                dealt=[]
                if not faceup:
                    for card in self.cards:
                         card.faceup=False
                
                for i in range (0,cards):
                        dealt.append(self.cards.pop())
                
                        
                for card in dealt:
                    
                    hand.add(card)

#__________________represents the overall game    

class Table(Hand):

    def __init__(self):

                
        self.cards=[]
        self.players=[]
        self.is_folded=False
        self.button=0
        self.hands=0
        self.blinds_timer=0
        
    def print_cards(self):
        rep='Community cards_______________\n'

        if self.is_folded:
            rep='FF'

        else:

            for card in self.cards:
                card.faceup=True
                rep+=str(card)+' '

        print (rep)

    def print_players(self):
        
        for player in self.players:
            print (player)
            
    def clear(self):

      self.cards=[]
      
      
      

#_______________POT represents the pot for each individual round of play

class Pot(object):
    
    stage_dict={0:'pre-flop bet', 1:'dealing the flop', 2:'dealing the turn', 3:'dealing the river'}
    deal_sequence=[0,3,1,1]
    pot_number=0
    
    def __init__(self, table, name):
        
   
        self.players=[]
        self.folded_players=[]
        self.active_players=[]
        self.limpers=0
        self.name=name
        self.blinds=BLINDS
                    
        self.total=0
        
        self.button=table.button
        #the amount each player has to call
        self.to_play=0
        #0=antes+ pre-flop, 1=post-flop, 2=turn, 3=river
        self.stage=0
        #defines turn within each betting stage
        self.turn=0
        #self.no_raise
        self.no_raise=0
        #already bet - works out if the round starts with 0 bet 
        self.already_bet=False
        self.raised=False
        

    @property

    def is_frozen(self):

        if len(self.active_players)<=1:
            self.active_players=[]
            return True
        else:
            return False

    @property

    def yet_to_play(self):

        ytp=self.table_size-(self.turn+1)
        if ytp<1: ytp=1

        return ytp

    @property

    def one_remaining(self):

        if len(self.folded_players)==(self.table_size-1):

            return True

        else:

            return False
        
    @property
    
    def table_size(self):
        
        
        return len(self.players)
        
    def __str__(self):

            rep='Pot= '+str(self.total)+'.  to play:'+str(self.to_play)
            return rep
            
    def set_blinds(self):
        
        dealer=(self.button)%self.table_size
        
        small_blind=(self.button+1)%self.table_size

        big_blind=(self.button+2)%self.table_size

        self.players[dealer].dealer=True

        self.players[small_blind].small_blind=True

        self.players[big_blind].big_blind=True

        return
        

    @property

    def who_plays(self):

        next_up=0

        if self.stage==0:

            next_up=(self.button+3)%self.table_size

            return next_up

        else:

            next_up=(self.button+1)%self.table_size
            return next_up


class Side_pot(Pot):
    
    serial=0
    
    def __init__(self, parent):
        
        Pot.__init__(self, parent, Pot)
        
        self.button=parent.button
        Side_pot.serial+=1
        self.name='side pot '+str(Side_pot.serial)
        
        self.players=[]
           


table=Table()

players = {}

# https://github.com/Pithikos/python-websocket-server/blob/master/server.py

# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
    for i in players.values():
        if (i[1] == client['id']):
            print(table.players)
            print(players)
            print(i[0].name)
            #table.players.remove(i[0])
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    print(message)
    if len(message) > 200:
        message = message[:200]+'..' 

    m = message.split(":")
    
    if(m[0] == 'join'):
        print("new player joined the table")
        players[m[1]] = (Hand(m[1], table, server, 'Human'), client)
    
    if(m[1] == 'turn'):
        cur = players[m[1]][0]
        
    #print("Client(%d) said: %s" % (client['id'], message))





PORT=24
s = WebsocketServer(PORT)
s.set_fn_new_client(new_client)
s.set_fn_client_left(client_left)
s.set_fn_message_received(message_received)

sockThread = Thread(target=s.run_forever, name="WebSocket")
sockThread.daemon = True
sockThread.start()



#________________FUNCTIONS____________________________________________________

#clears the players hands, comm cards, deck and moves the button on

def debug(pot):

    print('debug______________________')
    for player in pot.players:
        
        print (str(player.name)+' Stack='+str(player.stack)+' Stake='+str(player.stake)+' Player in pot='+str(player.in_pot)+'  Pot total='+str(pot.total)+'  all_in='+str(player.all_in)+'first all in'+str(player.first_all_in))
        print ('is folded'+str(player.is_folded))
        print ('raw data='+str(player.raw_data))
        print ('position='+str(player.position))
    
    
    for pot in pots:
            print (str(pot.name)+' total '+ str(pot.total))
            print ('yet to play:'+str(pot.yet_to_play))
            print ('active players')
            for player in pot.active_players:
                print (str(player.name))

            print ('table size '+str(pot.table_size))
            print ('limpers='+str(pot.limpers))
            print ('no raise '+str(pot.no_raise))
            print ('frozen='+str(pot.is_frozen))
            print ('one remaining='+str(pot.one_remaining))
            print ('Pot to play:  '+str(pot.to_play))
    print ('turn'+str(pot.turn)+'  no_raise'+str(pot.no_raise))
    print ('______________________________')

#helper function to move the play on

def next_player(pot, is_raise=False):

    pot.turn+=1
    if is_raise:
        pot.no_raise=1
    else:
        pot.no_raise+=1
        
    return


def next_hand(table, deck):

    table.clear()
    
    deck.clear()
    
    Side_pot.serial=0

    for hand in table.players:
        hand.clear()
        hand.small_blind=False
        hand.big_blind=False
        hand.dealer=False
        hand.first_all_in=False

    table.button+=1
    

#calculates the values and payouts

def ante_up(pot):

    for player in pot.players:
                
        player.ante(pot)
        print (player)
        deck.deal_to(player, 2)
        if player.stratname=='Human':
            player.flip()
        player.print_cards()
        pot.already_bet=True

    print (pot)
    print ('\n\n\n')


def betting_round(pot, table):

    global pots
    
    is_side_pot=False
    create_side_pot=False
    side_potters=[]
    
    while pot.no_raise<(pot.table_size):
        
                
        next_up=(int(pot.who_plays)+(pot.turn))%pot.table_size
        player=pot.players[next_up]
        player.to_play=(pots[-1].to_play-player.in_pot)
        if player.to_play<0:
            player.to_play=0
        
        

        #is the player folded? decide action

        if pots[-1].is_frozen==False:

            if player in pots[-1].active_players:
                  
                print (str(player.name)+' to play '+ str(player.to_play)+'\n')

                for strategy in player.strategy:
                    # Get player move and update all clients
                    if(player.stratname == 'Human'):
                        player.get_value()
        
                        # options=[['x', 'f', 'b'], ['c', 'r', 'f'], ['c', 'f']]
                        # choices={0:'check, fold or bet', 1:'call, raise, fold', 2:'call all-in or fold'}
                        # action=''
                        op=0


                        if player.to_play==0:
                                op=0
                        elif player.to_play<player.stack:
                                op=1
                        else: op=2


                        s.send_message_to_all(str(player.name)+' to play '+ str(player.to_play)+'\n')
                        
                        s.send_message(players[str(player.name)][1], str(op))
                        c = 30
                        while c > -1:
                            s.send_message_to_all("time:"+str(player.name)+":"+str(c))
                            time.sleep(1)
                            c-=1
                        player.no_play(pot)
                    else:
                        s.send_message_to_all(str(player.name)+' to play '+ str(player.to_play)+'\n')
                        strategy.decide_play(player, pots[-1])

                    

            else:
                
                player.no_play(pot)
            
        else:
            player.no_play(pot)
                         


#adjust player totals and check for all-ins
        
        pots[-1].total+=player.stake
        player.in_pot+=player.stake
        player.stack-=player.stake
         
        if player.stack==0 and player.first_all_in==False:
            
            print (str(player.name)+' is all in ')
            
            is_side_pot=True
            player.all_in=True
            player.first_all_in=True
            
            
    
        
        
        #debug(pot)
        
        
    if pots[-1].one_remaining:
        is_side_pot=False
            
    #deal with refunds_____________________________
                
    if is_side_pot:
        
        for player in pots[-1].players:
            if player.is_folded==False:
                
                side_potters.append(player)
            
        side_potters.sort(key=attrgetter('in_pot'), reverse=True)
        big_bet=side_potters[0].in_pot
        

        next_pot_players=[]
    
        
        
    #main pot refund_____________________
        
        print ('side pot')
        print ('high bet'+str(big_bet))
        low_bet=side_potters[-1].in_pot
        print ('low bet'+str(low_bet))
        
        for player in side_potters:
            
               
            refund=(player.in_pot-low_bet)
            if len(next_pot_players)>1:
                create_side_pot=True

            player.in_pot-=refund
            pot.total-=refund
            player.stack+=refund
            player.carry_over=refund

            print ('player in side pot - '+str(player.name))
            
            if player.carry_over>0:
                next_pot_players.append(player)
            else:
                if player in pots[-1].active_players:
                    pots[-1].active_players.remove(player)
      
            
            print (str(player.name))
            print ('refund...'+str(refund))

#create side pots _________________________________________________

        if create_side_pot:

            sidepot=Side_pot(pot)
            
            for player in next_pot_players:
                
                sidepot.players.append(player)
                
                sidepot.total+=player.carry_over
                player.in_pot+=player.carry_over
                player.stack-=player.carry_over
                
                if player.stack>0:
                    player.first_all_in=False
                    player.all_in=False
                    pots[-1].active_players.append(player)
                
                
                
            pots.append(sidepot)
            
                   

 #print out pot totals at the end of the round______________________________
       
    for pot in pots:
        print (str(pot.name))
        pot.to_play=0
        
        print ('pot size= '+str(pot.total))

#zero the player cash in the pot
        
        for player in pot.players:
            player.in_pot=0
            player.stake=0
            player.raised=0
        
  
#reset various pot variables for next betting round

    pots[0].no_raise=0
    pots[0].to_play=0
    pots[0].turn=0
    pots[0].stage+=1
    pots[0].already_bet=False
    pots[0].limpers=0
    
    
 


def showdown(pot):
        
    scoring=[]

    
    if pot.one_remaining:
        for player in pot.players:
            if player.is_folded==False:
                
                print (str(player.name)+' wins'+str(pot.total))
                player.stack+=pot.total
            

    else:

        for player in pot.players:
            if player.is_folded==False:
                player.get_value()
                scoring.append(player)
                 
                 
                 
        #rank hands in value+tie break order
                 
        scoring.sort(key=attrgetter('hand_value', 'tie_break'), reverse=True)
        split_pot=[]
        print ('\n\n\n')
        for player in scoring:

                if player.stratname!='Human':
                    player.flip()
                player.print_cards()
                print (player.name+' has '+str(player.rep))
                
                
        #check for split pot

        split_stake=0
        split=False
        
        for player in scoring[1:]:
            
            if player.hand_value==scoring[0].hand_value and player.tie_break==scoring[0].tie_break:

                split=True
                split_pot.append(scoring[0])
                split_pot.append(player)


        if split:

            print ('split pot')
            
            
            split_stake=int((pot.total/(len(split_pot))))
            for player in split_pot:
                     print (str(player.name)+' wins '+str(split_stake))
                     player.stack+=split_stake
                     
        else:
                
                scoring[0].stack+=pot.total
                print (str(scoring[0].name)+' wins '+str(pot.total))
        
        

               
        

#_______________________________________________________________________gameplay
        ######################

#set up the game and players






status='setup'

BLINDS=[10,20]

player6=Hand('Alex', table, 'SklanskySys2')



deck=Deck()

while status == 'setup':
    if (players.keys().__len__()) == 1:
        status = 'play' 

status='play'

#for i in range (0,2):
pots=[]
while status=='play':

    #increment the table hand#

     
    print('hi')


    #shuffle the deck
    
    deck.populate()
    deck.shuffle()

    #create pot for this hand
    
    pot=Pot(table, 'main')
    
    
    
    for player in table.players:
            pot.players.append(player)
            pot.active_players.append(player)
            
    pots.append(pot)

    
    #allocate blinds and ante up

    pot.set_blinds()

    print ('Hand#'+str(table.hands))
    print ('Blinds: '+str(BLINDS))
    
    ante_up(pot)

    #debug(pot)
    #table.print_players()

    while pot.stage<4:
            
        deck.deal_to(table, Pot.deal_sequence[pot.stage], True)

        print (str(Pot.stage_dict[pot.stage]))
        
        table.print_cards()            
        
        betting_round(pots[-1], table)
        
        #table.print_players()
       

    
    if len(table.players)>1:

        for pot in pots:
        
            showdown(pot)
            
         
    
    table.hands+=1
    table.blinds_timer=table.hands%6
    if table.blinds_timer==5:
        BLINDS[:] = [x*2 for x in BLINDS]
        
    for player in table.players[:]:
            print (player.name, player.stack, BLINDS[1])
            if player.stack<=BLINDS[1]:
                
                player.bust()
                
    if len(table.players)==1:
        status='winner'
    
          
    print ('\n\n\n')
    
    next_hand(table, deck)
    
for player in table.players:
    
    print (str(player.name)+' wins the game')


print(debug(pot))

    



    






   



