"""
A Q-learning agent for a stochastic task environment
"""

import random
import math
import sys


class RL_Agent(object):

    def __init__(self, states, valid_actions, parameters):
        self.alpha = parameters["alpha"]
        self.epsilon = parameters["epsilon"]
        self.gamma = parameters["gamma"]
        self.Q0 = parameters["Q0"]

        self.states = states
        self.Qvalues = {}
        for state in states:
            for action in valid_actions(state):
                self.Qvalues[(state, action)] = parameters["Q0"]


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        self.alpha = alpha


    def choose_action(self, state, valid_actions):
        """ Choose an action using epsilon-greedy selection.

        Args:
            state (tuple): Current robot state.
            valid_actions (list): A list of possible actions.
        Returns:
            action (string): Action chosen from valid_actions.
        """
        if random.random() < self.epsilon:
            return valid_actions[random.randint(0, len(valid_actions) - 1)]

        possible_actions = []
        for a in valid_actions:
            possible_actions.append((self.Qvalues[(state, a)], a))

        best_action = max(possible_actions, key=lambda x: x[0])[1]
        return best_action
        


    def update(self, state, action, reward, successor, valid_actions):
        """ Update self.Qvalues for (state, action) given reward and successor.

        Args:
            state (tuple): Current robot state.
            action (string): Action taken at state.
            reward (float): Reward given for transition.
            successor (tuple): Successor state.
            valid_actions (list): A list of possible actions at successor state.
        """
        cur_Qvalue = self.Qvalues[(state, action)]
        epsilon = self.epsilon
        self.setEpsilon(0)
        next_action = self.choose_action(successor, valid_actions)
        if successor == None:
            next_Qvalue = 0
        else:
            next_Qvalue = self.Qvalues[(successor, next_action)]
        self.setEpsilon(epsilon)
        
        self.Qvalues[(state, action)] = cur_Qvalue + (self.alpha * \
        (reward + (self.gamma * next_Qvalue) - cur_Qvalue))
