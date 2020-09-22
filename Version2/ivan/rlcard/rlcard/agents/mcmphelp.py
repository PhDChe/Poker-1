def par_UCT(rootstate, rootnode, itermax):
    print('hi')
    for i in range(0):
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
            # print(rootnode.ChildrenToString()) 
            pass
        
        # determine general performance of hand

        return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move