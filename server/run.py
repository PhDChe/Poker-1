from poker import Hand, Table, Deck, Pot, Side_pot
import poker





status='setup'

BLINDS=[10,20]

table=Table()

player1=Hand('Philip', table, 'SklanskySys2')
player2=Hand('tom', table, '')
player2=Hand('Igor', table, 'SklanskySys2')
player3=Hand('Carol', table, 'SklanskySys2')
player4=Hand('Johnboy', table, 'SklanskySys2')
player5=Hand('Rob', table, 'SklanskySys2')
player6=Hand('Alex', table, 'SklanskySys2')
player7=Hand('Wynona', table, 'SklanskySys2')
player8=Hand('Timur', table, 'SklanskySys2')

deck=Deck()

status='play'

#for i in range (0,2):

while status=='play':

    #increment the table hand#

     
        
    

    #shuffle the deck
    
    deck.populate()
    deck.shuffle()

    #create pot for this hand
    pots=[]
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