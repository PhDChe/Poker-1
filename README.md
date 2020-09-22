# ignitionBot
 - *bot.py* - Scrapes table data from an online casino game and returns game state, basic statistics on the current state (naive win-percentage and hand-strength given current state).
 - *ivan/ivvan.py* - Train NF-MCTS
 - - inspired by: https://arxiv.org/abs/1903.09569 (Not a direct implementation)
 - *ivan/ivvanPlay.py* - Play against a buggy bot that 'learned' strategy through a simple MC-NFSP (Monte Carlo - Neural Ficitious Self Play) implementation.
 - - NOTE: The MCTS is bugged and, in the long run, favores all-ins (especially in the early game when there is a lot of uncertainty)
 - MCTS implemented with root parallelization
 
# Version2
 - Version2 attempts to implement the WU-UCT (https://arxiv.org/abs/1810.11755) algorithm to Poker/imperfect information games.
 - - python main.py --model WU-UCT --env-name Poker --max-episode-length 1000 --MCTS-max-depth 1000000 --MCTS-max-steps 10000 --MCTS-max-width 100000
 
# server
 - Backend for a websocket server and python game

# todo
 - Refrain from using MCTS in preflop (too little information)/implement a general pre-flop strategy
 - Add MCTS favored move to *bot.py* statistics 
