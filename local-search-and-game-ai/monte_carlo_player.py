#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
MCTS AI player for Othello.
"""

import random
import numpy as np
from six.moves import input
from othello_shared import get_possible_moves, play_move, compute_utility


class Node:
    def __init__(self, state, player, parent, children, v=0, N=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.children = children
        self.value = v
        self.N = N

    def get_child(self, state):
        for c in self.children:
            if (state == c.state).all():
                return c
        return None


def select(root, alpha):
    """ Starting from given node, find a terminal node or node with unexpanded children.
    If all children of a node are in tree, move to the one with the highest UCT value.

    Args:
        root (Node): MCTS tree root node
        alpha (float): Weight of exploration term in UCT

    Returns:
        node (Node): Node at bottom of MCTS tree
    """
    # TODO:
    moves = get_possible_moves(root.state, root.player)
    # Terminal Node
    if len(moves) == 0:
        return root
    # Successor not in tree
    if len(moves) > len(root.children):
        return root

    # Find child with highest UCT value
    highest_val = root.children[0]
    for child in root.children:
        UCT_child = child.value + alpha*((np.log(root.N) / child.N)**0.5)
        UCT_highest = highest_val.value + alpha*((np.log(root.N) / highest_val.N)**0.5)
        if UCT_child > UCT_highest:
            highest_val = child

    # Make child with highest UCT value the new root node
    return select(highest_val, alpha)
    # return root


def expand(node):
    """ Add a child node of state into the tree if it's not terminal.

    Args:
        node (Node): Node to expand

    Returns:
        leaf (Node): Newly created node (or given Node if already leaf)
    """
    # TODO:

    succs = get_possible_moves(node.state, node.player)
    for suc in succs:
        (i, j) = suc
        new_state = play_move(node.state, node.player, i, j)
        if node.get_child(new_state) == None:
            # Add new state to new node
            new_node = Node(new_state, 3 - node.player, node, [], 0, 0)
            node.children.append(new_node)

            return new_node
    
    return node


def simulate(node):
    """ Run one game rollout using from state to a terminal state.
    Use random playout policy.

    Args:
        node (Node): Leaf node from which to start rollout.

    Returns:
        utility (int): Utility of final state
    """
    # TODO:
    cur_state = node.state
    cur_player = node.player

    while True:
        moves = get_possible_moves(cur_state, cur_player)
    
        if len(moves) == 0:
            break
            
        # Pick a random move
        (i, j) = random.choice(moves)
    
        # Set new state, player
        cur_state = play_move(cur_state, cur_player, i, j)
        cur_player = 3 - cur_player
    
    return compute_utility(cur_state)


def backprop(node, utility):
    """ Backpropagate result from state up to the root.
    Every node has N, number of plays, incremented
    If node's parent is dark (1), then node's value increases
    Otherwise, node's value decreases.

    Args:
        node (Node): Leaf node from which rollout started.
        utility (int): Utility of simulated rollout.
    """
    # TODO:

    node.N += 1
    player_util = utility
    
    if node.player == 1:
        player_util = utility * -1

    node.value = (node.value * (node.N - 1) + player_util) / node.N

    # Done if root node
    if node.parent == None:
        return
    
    return backprop(node.parent, utility)


def mcts(state, player, rollouts=100, alpha=5):
    # MCTS main loop: Execute four steps rollouts number of times
    # Then return successor with highest number of rollouts
    root = Node(state, player, None, [], 0, 1)
    for i in range(rollouts):
        leaf = select(root, alpha)
        new = expand(leaf)
        utility = simulate(new)
        backprop(new, utility)

    move = None
    plays = 0
    for m in get_possible_moves(state, player):
        s = play_move(state, player, m[0], m[1])
        if root.get_child(s).N > plays:
            plays = root.get_child(s).N
            move = m

    return move


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("MCTS AI")        # First line is the name of this AI
    color = int(input())    # 1 for dark (first), 2 for light (second)

    while True:
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()

        if status == "FINAL":
            print()
        else:
            board = np.array(eval(input()))
            movei, movej = mcts(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()